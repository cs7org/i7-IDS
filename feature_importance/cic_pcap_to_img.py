from pathlib import Path
from loguru import logger
from scapy.all import Packet
import numpy as np
from scapy.all import IP, TCP, Ether, raw
import cv2
import json
from tqdm import tqdm
from scapy.utils import RawPcapReader
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP, Ether
import pickle
from ids_expt.core.defs import Session


def stream_pcap(pcap_path):
    """Generator to yield parsed packets and their timestamps from a pcap file"""
    for pkt_data, pkt_metadata in RawPcapReader(pcap_path):
        pkt = Ether(pkt_data)
        ts = pkt_metadata.sec + pkt_metadata.usec / 1e6
        yield pkt, ts


def is_dnp3_packet(pkt):
    # Check if packet has TCP or UDP layer
    if pkt.haslayer(TCP):
        l4 = pkt[TCP]
    elif pkt.haslayer(UDP):
        l4 = pkt[UDP]
    else:
        return False

    # Check if either src or dst port is 20000 (DNP3 default port)
    if l4.sport != 20000 and l4.dport != 20000:
        return False

    # Check if payload exists and starts with DNP3 header bytes 0x05 0x64
    raw = bytes(l4.payload)
    if len(raw) >= 2 and raw[0] == 0x05 and raw[1] == 0x64:
        return True
    return False


def anonymize_packet(packet: Packet) -> Packet:
    """Anonymize packet by removing address information"""
    # Create copy to avoid modifying original packet
    pkt = packet.copy()

    # IP layer handling
    if pkt.haslayer(IP):
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"

    # Ethernet layer handling
    if pkt.haslayer(Ether):
        pkt[Ether].src = "00:00:00:00:00:00"
        pkt[Ether].dst = "00:00:00:00:00:00"

    # TCP layer handling
    if pkt.haslayer(TCP):
        pkt[TCP].sport = 0
        pkt[TCP].dport = 0

    return pkt


class PCAPSessionFeatureExtractor:
    def __init__(
        self,
        pcap_path: Path,
        label_df: pd.DataFrame = pd.DataFrame(),
        out_dir: Path = Path("output"),
        max_packets: int = -1,
        max_bytes: int = -1,
        max_sessions: int = -1,
    ):
        self.out_dir = out_dir
        self.max_packets = max_packets
        self.max_bytes = max_bytes
        self.pcap_path = pcap_path
        self.packet_buffer = None
        self.out_dir = Path(out_dir) / pcap_path.stem
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = []
        self.stats = None
        self.max_sessions = max_sessions
        self.label_df = label_df

    def load(self):
        """Load packets from the PCAP file."""
        logger.info(f"Loading packets from {self.pcap_path}...")
        self.packet_buffer = list(stream_pcap(str(self.pcap_path)))
        logger.info(f"Loaded {len(self.packet_buffer)} packets from {self.pcap_path}")

    def packets_to_labelled_sessions(self, packet_buffer: list[tuple[Packet, float]]):
        df = self.label_df.copy()
        sessions = []
        # packets = [(p.copy(), ts) for p, ts in packet_buffer]
        df["Start_ts"] = df["Timestamp"].apply(lambda t: t.timestamp())
        df["End_ts"] = df["Start_ts"] + df["Flow Duration"] / 1e6
        df["EndTimestamp"] = df["Timestamp"] + pd.to_timedelta(
            df["Flow Duration"], unit="us"
        )
        packet_ts = np.array([pts[1] for pts in packet_buffer])
        completed_packets = []
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing Sessions"
        ):
            start_time = row["Timestamp"] - pd.to_timedelta(1, unit="s")
            end_time = row["EndTimestamp"] + pd.to_timedelta(1, unit="s")
            src_ip = row["Src IP"]
            dst_ip = row["Dst IP"]
            src_port = int(row["Src Port"])
            dst_port = int(row["Dst Port"])
            proto = int(row["Protocol"])
            label = row["Label"] if "Label" in row else "NORMAL"

            matched_packets = []
            raw_bytes = []

            start_ts = start_time.timestamp()
            end_ts = end_time.timestamp()

            # Find matching packets based on timestamps
            matching_idx = np.where((packet_ts >= start_ts) & (packet_ts <= end_ts))[0]
            unchecked_packets = [packet_buffer[i][0] for i in matching_idx]
            unchecked_timestamps = [packet_ts[i] for i in matching_idx]
            for pkt, pkt_ts in zip(unchecked_packets, unchecked_timestamps):
                added_packet = False
                if pkt.haslayer(IP) and pkt.haslayer(Ether):
                    ip_layer = pkt.getlayer(IP)
                    if (ip_layer.src == src_ip and ip_layer.dst == dst_ip) or (
                        ip_layer.src == dst_ip and ip_layer.dst == src_ip
                    ):
                        if (
                            pkt.haslayer(TCP)
                            and (
                                pkt.getlayer(TCP).sport == src_port
                                and pkt.getlayer(TCP).dport == dst_port
                            )
                        ) or (
                            pkt.haslayer(UDP)
                            and (
                                pkt.getlayer(UDP).sport == src_port
                                and pkt.getlayer(UDP).dport == dst_port
                            )
                        ):
                            matched_packets.append(pkt)
                            raw_bytes.append(pkt.original)
                            added_packet = True
                if not added_packet:
                    if is_dnp3_packet(pkt):
                        matched_packets.append(pkt)
                        raw_bytes.append(pkt.original)

            interval = (end_time - start_time).total_seconds()
            sessions.append(
                Session(
                    start_time=start_time,
                    end_time=end_time,
                    packets=matched_packets,
                    interval=interval,
                    raw_bytes=raw_bytes,
                    label=label,
                    flow_id=row["Flow ID"],
                )
            )

        stats = self.session_statistics(sessions)
        self.stats = stats
        return sessions, stats

    def extract_session_features(
        self, session_packets, max_packets=300, bytes_per_packet=128
    ):
        """
        Extract first N bytes from first M packets of a session

        Args:
            session_packets (list): List of packets in session
            max_packets (int): Maximum number of packets to process
            bytes_per_packet (int): Number of bytes to extract per packet

        Returns:
            tuple: (8x128 grayscale array, 8x128 byte sequence array)
        """
        # Initialize arrays
        grayscale_data = np.zeros((max_packets, bytes_per_packet), dtype=np.uint8)

        # Process up to max_packets
        processed_packets = 0
        for i, packet in enumerate(session_packets[:max_packets]):
            try:
                packet = anonymize_packet(packet)
                raw_bytes = raw(packet)

                # Extract first bytes_per_packet bytes
                packet_data = raw_bytes[:bytes_per_packet]

                # Pad if necessary
                if len(packet_data) < bytes_per_packet:
                    packet_data += b"\x00" * (bytes_per_packet - len(packet_data))

                # Convert to numpy array
                packet_array = np.frombuffer(packet_data, dtype=np.uint8)

                # Store in both formats
                grayscale_data[processed_packets] = packet_array

                processed_packets += 1

            except Exception as e:
                print(f"Error processing packet {i}: {e}")
                continue

        return grayscale_data

    def extract_sessions(self):
        if not self.packet_buffer:
            logger.warning("No packets found in the PCAP file.")
            return
        logger.info(f"Extracting sessions from {len(self.packet_buffer)} packets.")

        logger.info(f"Processing interval: {self.interval} seconds")
        sessions = self.packets_to_sessions(
            self.packet_buffer, interval_threshold=self.interval
        )
        for session_packets in sessions:
            if not session_packets:
                continue
            start_time = session_packets[0].time
            end_time = session_packets[-1].time
            raw_bytes = [raw(pkt) for pkt in session_packets]
            # array = np.array([bytes(pkt) for pkt in session_packets])

            session = Session(
                start_time=start_time,
                end_time=end_time,
                packets=session_packets,
                interval=self.interval,
                raw_bytes=raw_bytes,
                # array=array,
            )
            self.sessions.append(session)
            # logger.info(f"Created session: {session}")
        stats = self.session_statistics(self.sessions)
        self.stats = stats
        return self.sessions, stats

    def session_statistics(self, sessions: list[Session]):
        num_sessions = len([s for s in sessions if len(s.packets) > 0])
        num_packets = sum(sess.num_packets for sess in sessions)
        num_bytes = [len(b) for sess in sessions for b in sess.raw_bytes]
        max_pkt_count = max(sess.num_packets for sess in sessions)
        avg_packet_count = np.mean([sess.num_packets for sess in sessions])
        avg_byter_per_pkt = np.mean(num_bytes) if num_bytes else 0
        max_byte_per_pkt = max(num_bytes) if num_bytes else 0
        stats = {
            "labelled_sessions": num_sessions,
            "labelled_packets": num_packets,
            "avg_packet_count": avg_packet_count,
            "max_pkt_count": max_pkt_count,
            "avg_bytes_per_packet": avg_byter_per_pkt,
            "max_bytes_per_packet": max_byte_per_pkt,
        }
        logger.info(f"Session statistics: {stats}")

        return stats

    def sessions_to_image(self, sessions: list[Session]):
        """Convert sessions to grayscale images and save them."""
        max_bytes = (
            self.stats.get("max_bytes_per_packet", 128)
            if self.max_bytes < 0
            else self.max_bytes
        )
        max_packets = (
            self.stats.get("max_pkt_count", 300)
            if self.max_packets < 0
            else self.max_packets
        )
        i = 0
        for session in tqdm(sessions, desc="Processing sessions", unit="session"):
            # Skip empty sessions
            if not session.packets:
                continue

            image_dir = self.out_dir / f"session_{i}_{session.label}.png"

            # Extract features
            grayscale_array = self.extract_session_features(
                session.packets, max_packets=max_packets, bytes_per_packet=max_bytes
            )
            cv2.imwrite(str(image_dir), grayscale_array)
            # hot image no need! do it later and save memory
            # fig = plt.figure(figsize=(10, 10))
            # plt.imshow(grayscale_array, cmap="hot", aspect="auto")
            # plt.colorbar()
            # plt.axis("off")
            # plt.savefig(
            #     str(image_dir.parent / f"{image_dir.stem}_hot.png"),
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # plt.close(fig)
            # logger.info(f"Saved session {i} image to {image_dir}")
            i += 1
        logger.info(f"Saved {i} session images to {self.out_dir}")

    def run(self, labelled: bool = False):
        """Run the feature extraction and session processing."""
        # Load packets from PCAP file
        if self.packet_buffer is None:
            self.load()
        else:
            logger.info("Packets already loaded. Skipping load step.")
        if not self.packet_buffer:
            logger.error("No packets loaded. Exiting.")
            return
        logger.info("Starting feature extraction...")

        num_packets = len(self.packet_buffer)
        if labelled:
            logger.info("Processing labelled sessions...")
            self.sessions, stats = self.packets_to_labelled_sessions(self.packet_buffer)
        else:
            logger.info("Processing unlabelled sessions...")
            self.sessions, stats = self.extract_sessions()
        if not self.sessions:
            logger.warning("No sessions extracted. Exiting.")
            return

        logger.info("Extracted sessions successfully.")

        labelled_packets = sum(len(sess.packets) for sess in self.sessions)
        logger.info(
            f"Total sessions: {len(self.sessions)}, Total packets: {len(self.packet_buffer)}, Labelled packets: {labelled_packets}"
        )
        num_sessions = len(self.label_df)
        stats["num_packets"] = num_packets
        stats["num_sessions"] = num_sessions
        stats_file = self.out_dir / "session_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved session statistics to {stats_file}")

        # Save session images
        self.sessions_to_image(self.sessions)

        # save sessions to pickle
        sessions_file = self.out_dir / "sessions.pkl"
        with open(sessions_file, "wb") as f:
            pickle.dump(self.sessions, f)

        logger.info("Feature extraction completed successfully.")


if __name__ == "__main__":
    import json

    map_file = Path(r"E:\MSc Works\IDS\notebooks\cic_file_mapping.json")
    pcap_root = Path(r"E:\MSc Works\IDS\data\Total PCAP Files")
    csv_root = Path(r"E:\MSc Works\IDS\data\CICFlowMeter")

    with open(map_file, "r") as f:
        file_mapping = json.load(f)

    files = [f for f in csv_root.glob("*.csv") if f.is_file()]
    files_sorted_by_size = sorted(files, key=lambda f: f.stat().st_size)

    for idx, cfile in enumerate(files_sorted_by_size):
        fname = cfile.name
        if fname not in file_mapping:
            print(f"Skipping {fname} as no matching pcap file found.")
            continue
        df = pd.read_csv(cfile)

        logger.info(f"Loaded CSV from {cfile}")
        df.columns = [c.strip() for c in df.columns]
        duplicates = df.columns.duplicated(keep="last")
        df = df.loc[:, ~duplicates]
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        pcap_session_extractor = PCAPSessionFeatureExtractor(
            pcap_path=pcap_root / file_mapping[fname],
            label_df=df,
            out_dir=Path("output_updated"),
        )
        pcap_session_extractor.run(labelled=True)
        logger.info(
            f"Processed {idx}/{len(files)}: {fname} with {len(pcap_session_extractor.sessions)} sessions."
        )
        logger.info(f"Session statistics: {pcap_session_extractor.stats}")
        # break
