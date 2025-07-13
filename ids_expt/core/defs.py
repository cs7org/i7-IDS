from enum import Enum
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Session:
    flow_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    packets: list["Packet"]
    interval: float
    raw_bytes: list[bytearray]
    label: str = "NORMAL"
    num_forward_packets: int = 0
    num_backward_packets: int = 0
    expected_forward_packets: int = 0
    expected_backward_packets: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_packets(self) -> int:
        return len(self.packets)

    def __repr__(self):
        return (
            f"Session(start_time={self.start_time}, end_time={self.end_time}, "
            f"num_packets={self.num_packets}, interval={self.interval})"
        )


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"


class DataType(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"

    def __str__(self) -> str:
        return self.value


class SamplingMethod(str, Enum):
    OVERSAMPLE = "oversample"
    UNDERSAMPLE = "undersample"
    NONE = "nosampling"


class NormalizationMethod(str, Enum):
    MIN_MAX = "min_max"
    STANDARD = "standard"


class MetricType(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LOSS = "loss"


TOP_FEATURES: list[str] = [
    "destination port",
    "TRpktLenVAR",
    "APPpktLenSTD",
    "TRpktLenMAX",
    "TotPktsInFlow",
    "APPpktLenMEAN",
    "APPpktLenVAR",
    "TRpktLenMEAN",
    "TRpktLenSTD",
    "DLpktLenVAR",
    "DLpktLenMAX",
    "FlowIAT_MEAN",
    "DLpktLenMEAN",
    "TotalFwdPkts",
    "DLpktLenSTD",
    "APPpktLenMAX",
    "DLfwdPktLenMIN",
    "pktsFromSLAVE",
    "TRfwdPktLenMIN",
    "duration",
    "APPflowBytes/sec",
    "DLflowBytes/sec",
    "TRfwdHdrLen",
    "pktsFromMASTER",
    "TRflowBytes/sec",
    "DLfwdHdrLen",
    "mostCommonREQ_FUNC_CODE",
    "FlowIAT_STD",
    "TRbwdHdrLen",
    "APPbwdHdrLen",
    "TotalFwdIAT",
    "FlowIAT_MAX",
    "TotLenbwdTR",
    "mostCommonRESP_FUNC_CODE",
    "DLpktLenMIN",
    "bwdPkts/sec",
    "DLbwdHdrLen",
    "APPfwdPktLenSTD",
    "bwdIAT_MEAN",
    "fwdPkts/sec",
    "fwdIAT_MEAN",
    "FlowPkts/sec",
    "TRfwdPktLenSTD",
    "TotalBwdPkts",
    "TRbwdPktLenMAX",
    "source port",
]

TOP_CIC_FEATURES = [
    "Subflow Bwd Pkts",
    "Bwd Header Len",
    "Fwd Pkt Len Min",
    "Tot Bwd Pkts",
    "Subflow Bwd Byts",
    "Tot Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd IAT Mean",
    "Fwd Act Data Pkts",
    "Fwd Header Len",
    "Subflow Fwd Byts",
    "Fwd Pkt Len Std",
    "Fwd Pkts/s",
    "TotLen Fwd Pkts",
    "Fwd Seg Size Avg",
    "Flow IAT Std",
    "Flow Pkts/s",
    "Flow Byts/s",
    "Pkt Len Mean",
    "Fwd Pkt Len Mean",
    "Pkt Size Avg",
    "Flow IAT Mean",
    "Subflow Fwd Pkts",
    "Bwd IAT Tot",
    "Pkt Len Var",
    "Bwd Pkt Len Mean",
    "Fwd IAT Std",
    "Bwd IAT Mean",
    "Init Bwd Win Byts",
    "Dst Port",
    "Fwd IAT Tot",
    "Bwd IAT Std",
    "Bwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkts/s",
    "Flow Duration",
    "Src Port",
]
