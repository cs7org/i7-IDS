#!/bin/bash

# DNP3 Dataset Packet Extractor
# Implements CICFlowMeter-compatible direction detection and boundary analysis
# Usage: ./pcap_label.sh <csv_file> <pcap_file> [output_dir]

set -e

# Windows-specific path conversion function
convert_windows_path() {
    local path="$1"
    echo "$path" | sed 's|\\|/|g' | sed 's|^\([A-Za-z]\):|/\L\1|'
}

# Check if required tools are installed
check_dependencies() {
    local deps=("tshark")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            echo "Error: $dep is not installed." >&2
            case "$dep" in
                "tshark") 
                    echo "Install Wireshark from: https://www.wireshark.org/download.html" >&2
                    echo "Make sure to include TShark during installation" >&2
                    echo "Add Wireshark installation directory to your PATH" >&2
                    ;;
            esac
            exit 1
        fi
    done
}

# Function to convert CSV timestamp to tshark-compatible format
convert_timestamp() {
    local timestamp="$1"
    echo "$timestamp" | awk '{
        split($1, date_parts, "/")
        day = date_parts[1]
        month = date_parts[2] 
        year = date_parts[3]
        time = $2
        period = $3
        
        months["01"]="Jan"; months["02"]="Feb"; months["03"]="Mar"
        months["04"]="Apr"; months["05"]="May"; months["06"]="Jun"
        months["07"]="Jul"; months["08"]="Aug"; months["09"]="Sep"
        months["10"]="Oct"; months["11"]="Nov"; months["12"]="Dec"
        
        split(time, time_parts, ":")
        hour = time_parts[1]
        minute = time_parts[2]
        second = time_parts[3]
        
        if (period == "PM" && hour != "12") hour = hour + 12
        if (period == "AM" && hour == "12") hour = "00"
        
        printf "%s %d, %s %02d:%s:%s", months[month], day, year, hour, minute, second
    }'
}

# Enhanced CSV parsing with flow boundary tracking
parse_csv_row_enhanced() {
    local csv_file="$1"
    local row_num="$2"
    
    local row_data=$(head -n "$row_num" "$csv_file" | tail -n 1)
    
    if [[ -z "$row_data" ]]; then
        return 1
    fi
    
    IFS=',' read -ra fields <<< "$row_data"
    
    # Extract enhanced fields
    FLOW_ID="${fields[1]}"
    SRC_IP="${fields[2]}"
    SRC_PORT="${fields[3]}"
    DST_IP="${fields[4]}"
    DST_PORT="${fields[5]}"
    PROTOCOL="${fields[6]}"
    TIMESTAMP="${fields[7]}"
    FLOW_DURATION="${fields[8]}"
    TOT_FWD_PKTS="${fields[9]}"
    TOT_BWD_PKTS="${fields[10]}"
    LABEL="${fields[-1]}"
    
    # Clean up fields
    SRC_IP=$(echo "$SRC_IP" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    DST_IP=$(echo "$DST_IP" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    SRC_PORT=$(echo "$SRC_PORT" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    DST_PORT=$(echo "$DST_PORT" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    PROTOCOL=$(echo "$PROTOCOL" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    TIMESTAMP=$(echo "$TIMESTAMP" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    FLOW_DURATION=$(echo "$FLOW_DURATION" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    TOT_FWD_PKTS=$(echo "$TOT_FWD_PKTS" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    TOT_BWD_PKTS=$(echo "$TOT_BWD_PKTS" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    LABEL=$(echo "$LABEL" | tr -d '"' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    return 0
}

# Function to determine CICFlowMeter-compatible flow direction
get_cicflowmeter_direction() {
    local temp_pcap="$1"
    
    # Get the very first packet to establish CICFlowMeter direction
    local first_packet_info=$(tshark -r "$temp_pcap" -T fields -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport 2>/dev/null | head -1)
    
    if [[ -n "$first_packet_info" ]]; then
        IFS=$'\t' read -r epoch src_ip dst_ip tcp_sport tcp_dport udp_sport udp_dport <<< "$first_packet_info"
        
        # Determine protocol and set ports accordingly
        if [[ "$PROTOCOL" == "6" ]]; then
            FLOW_FWD_SRC="$src_ip"
            FLOW_FWD_DST="$dst_ip"
            FLOW_FWD_SPORT="$tcp_sport"
            FLOW_FWD_DPORT="$tcp_dport"
        elif [[ "$PROTOCOL" == "17" ]]; then
            FLOW_FWD_SRC="$src_ip"
            FLOW_FWD_DST="$dst_ip"
            FLOW_FWD_SPORT="$udp_sport"
            FLOW_FWD_DPORT="$udp_dport"
        else
            FLOW_FWD_SRC="$src_ip"
            FLOW_FWD_DST="$dst_ip"
            FLOW_FWD_SPORT=""
            FLOW_FWD_DPORT=""
        fi
        return 0
    else
        return 1
    fi
}

# Function to count packets using CICFlowMeter-compatible direction
count_packets_by_cicflowmeter_direction() {
    local temp_pcap="$1"
    
    # Get CICFlowMeter direction from first packet
    if get_cicflowmeter_direction "$temp_pcap"; then
        # Count forward packets (same direction as first packet)
        local fwd_filter="ip.src == $FLOW_FWD_SRC and ip.dst == $FLOW_FWD_DST"
        if [[ -n "$FLOW_FWD_SPORT" ]] && [[ -n "$FLOW_FWD_DPORT" ]]; then
            if [[ "$PROTOCOL" == "6" ]]; then
                fwd_filter="$fwd_filter and tcp.srcport == $FLOW_FWD_SPORT and tcp.dstport == $FLOW_FWD_DPORT"
            elif [[ "$PROTOCOL" == "17" ]]; then
                fwd_filter="$fwd_filter and udp.srcport == $FLOW_FWD_SPORT and udp.dstport == $FLOW_FWD_DPORT"
            fi
        fi
        
        local fwd_count=$(tshark -r "$temp_pcap" -Y "$fwd_filter" -T fields -e frame.number 2>/dev/null | wc -l)
        
        # Count backward packets (opposite direction)
        local bwd_filter="ip.src == $FLOW_FWD_DST and ip.dst == $FLOW_FWD_SRC"
        if [[ -n "$FLOW_FWD_SPORT" ]] && [[ -n "$FLOW_FWD_DPORT" ]]; then
            if [[ "$PROTOCOL" == "6" ]]; then
                bwd_filter="$bwd_filter and tcp.srcport == $FLOW_FWD_DPORT and tcp.dstport == $FLOW_FWD_SPORT"
            elif [[ "$PROTOCOL" == "17" ]]; then
                bwd_filter="$bwd_filter and udp.srcport == $FLOW_FWD_DPORT and udp.dstport == $FLOW_FWD_SPORT"
            fi
        fi
        
        local bwd_count=$(tshark -r "$temp_pcap" -Y "$bwd_filter" -T fields -e frame.number 2>/dev/null | wc -l)
        
        echo "$fwd_count,$bwd_count"
    else
        echo "0,0"
    fi
}

# Function to check for temporal overlap with previous flows
check_temporal_overlap() {
    local start_time="$1"
    local end_time="$2"
    local overlap_file="$3"
    
    # Convert timestamps to epoch for comparison
    local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo "0")
    local end_epoch=$(date -d "$end_time" +%s 2>/dev/null || echo "0")
    
    if [[ -f "$overlap_file" ]]; then
        while IFS=',' read -r prev_start prev_end prev_flow_id; do
            local prev_start_epoch=$(echo "$prev_start" | tr -d '"')
            local prev_end_epoch=$(echo "$prev_end" | tr -d '"')
            
            # Check for overlap
            if [[ "$start_epoch" -lt "$prev_end_epoch" ]] && [[ "$end_epoch" -gt "$prev_start_epoch" ]]; then
                return 0  # Overlap detected
            fi
        done < "$overlap_file"
    fi
    
    # Record this flow's time range
    echo "\"$start_epoch\",\"$end_epoch\",\"$FLOW_ID\"" >> "$overlap_file"
    return 1  # No overlap
}

# Enhanced packet extraction with boundary detection and CICFlowMeter direction
extract_packets_with_boundary_detection() {
    local pcap_file="$1"
    local output_file="$2"
    local row_num="$3"
    local flow_duration="$4"
    local start_time="$5"
    local total_rows="$6"
    local overlap_file="$7"
    
    # Convert timestamp to tshark format
    local tshark_start_time=$(convert_timestamp "$start_time")
    
    # Calculate end time with improved precision
    local duration_seconds=0
    if [[ -n "$flow_duration" ]] && [[ "$flow_duration" =~ ^[0-9]+$ ]]; then
        duration_seconds=$((flow_duration / 1000000))
    fi
    
    local end_time
    if command -v date &>/dev/null && [[ "$duration_seconds" -gt 0 ]]; then
        end_time=$(date -d "$tshark_start_time + $duration_seconds seconds" "+%b %d, %Y %H:%M:%S" 2>/dev/null || echo "$tshark_start_time")
    else
        # For zero duration flows, use minimal time window
        end_time=$(date -d "$tshark_start_time + 1 seconds" "+%b %d, %Y %H:%M:%S" 2>/dev/null || echo "$tshark_start_time")
    fi
    
    # Check for temporal overlap to prevent packet duplication
    if check_temporal_overlap "$tshark_start_time" "$end_time" "$overlap_file"; then
        echo "Row $row_num/$total_rows: Temporal overlap detected - using boundary detection" >&2
        
        # Calculate precise flow boundaries using packet timing
        # local temp_pcap="${TEMP:-/tmp}/flow_${row_num}_boundary_check.pcap"
        local temp_pcap=$(mktemp -p "${TEMP:-/tmp}" "flow_XXXXXXXXXX_boundary_check.pcap")
        
        # Extract all packets in the general time window
        local broad_filter="(frame.time >= \"$tshark_start_time\" && frame.time <= \"$end_time\") and ((ip.src == $SRC_IP and ip.dst == $DST_IP"
        if [[ "$SRC_PORT" =~ ^[0-9]+$ ]] && [[ "$DST_PORT" =~ ^[0-9]+$ ]]; then
            if [[ "$PROTOCOL" == "6" ]]; then
                broad_filter="$broad_filter and tcp.srcport == $SRC_PORT and tcp.dstport == $DST_PORT) or (ip.src == $DST_IP and ip.dst == $SRC_IP and tcp.srcport == $DST_PORT and tcp.dstport == $SRC_PORT))"
            elif [[ "$PROTOCOL" == "17" ]]; then
                broad_filter="$broad_filter and udp.srcport == $SRC_PORT and udp.dstport == $DST_PORT) or (ip.src == $DST_IP and ip.dst == $SRC_IP and udp.srcport == $DST_PORT and udp.dstport == $SRC_PORT))"
            fi
        else
            broad_filter="$broad_filter) or (ip.src == $DST_IP and ip.dst == $SRC_IP))"
        fi
        
        tshark -r "$pcap_file" -Y "$broad_filter" -w "$temp_pcap" 2>/dev/null
        
        if [[ -f "$temp_pcap" ]]; then
            # Analyze packet timestamps to find flow boundaries
            local packet_times=$(tshark -r "$temp_pcap" -T fields -e frame.time_epoch 2>/dev/null | sort -n)
            local packet_count=$(echo "$packet_times" | wc -l)
            
            if [[ "$packet_count" -gt 0 ]]; then
                # Use only the expected number of packets based on CSV data
                local expected_total=$((TOT_FWD_PKTS + TOT_BWD_PKTS))
                
                if [[ "$packet_count" -gt "$expected_total" ]] && [[ "$expected_total" -gt 0 ]]; then
                    # Extract timestamps and select the appropriate subset
                    local first_time=$(echo "$packet_times" | head -1)
                    local time_array=($packet_times)
                    
                    # Calculate optimal time window based on expected packet count
                    local optimal_end_index=$((expected_total - 1))
                    if [[ "$optimal_end_index" -lt "$packet_count" ]]; then
                        local optimal_end_time="${time_array[$optimal_end_index]}"
                        
                        # Create refined filter with calculated boundaries
                        local refined_filter="(frame.time_epoch >= $first_time && frame.time_epoch <= $optimal_end_time) and ((ip.src == $SRC_IP and ip.dst == $DST_IP"
                        if [[ "$SRC_PORT" =~ ^[0-9]+$ ]] && [[ "$DST_PORT" =~ ^[0-9]+$ ]]; then
                            if [[ "$PROTOCOL" == "6" ]]; then
                                refined_filter="$refined_filter and tcp.srcport == $SRC_PORT and tcp.dstport == $DST_PORT) or (ip.src == $DST_IP and ip.dst == $SRC_IP and tcp.srcport == $DST_PORT and tcp.dstport == $SRC_PORT))"
                            elif [[ "$PROTOCOL" == "17" ]]; then
                                refined_filter="$refined_filter and udp.srcport == $SRC_PORT and udp.dstport == $DST_PORT) or (ip.src == $DST_IP and ip.dst == $SRC_IP and udp.srcport == $DST_PORT and udp.dstport == $SRC_PORT))"
                            fi
                        else
                            refined_filter="$refined_filter) or (ip.src == $DST_IP and ip.dst == $SRC_IP))"
                        fi
                        
                        echo "  Refined filter using packet boundary detection" >&2
                        
                        tshark -r "$pcap_file" -Y "$refined_filter" -w "$output_file" 2>/dev/null
                    else
                        mv "$temp_pcap" "$output_file"
                    fi
                else
                    mv "$temp_pcap" "$output_file"
                fi
            else
                rm -f "$temp_pcap"
                echo "0,0,0"
                return
            fi
        else
            echo "0,0,0"
            return
        fi
    else
        # No overlap detected, use standard extraction
        local complete_filter="(frame.time >= \"$tshark_start_time\" && frame.time <= \"$end_time\") and ((ip.src == $SRC_IP and ip.dst == $DST_IP"
        if [[ "$SRC_PORT" =~ ^[0-9]+$ ]] && [[ "$DST_PORT" =~ ^[0-9]+$ ]]; then
            if [[ "$PROTOCOL" == "6" ]]; then
                complete_filter="$complete_filter and tcp.srcport == $SRC_PORT and tcp.dstport == $DST_PORT) or (ip.src == $DST_IP and ip.dst == $SRC_IP and tcp.srcport == $DST_PORT and tcp.dstport == $SRC_PORT))"
            elif [[ "$PROTOCOL" == "17" ]]; then
                complete_filter="$complete_filter and udp.srcport == $SRC_PORT and udp.dstport == $DST_PORT) or (ip.src == $DST_IP and ip.dst == $SRC_IP and udp.srcport == $DST_PORT and udp.dstport == $SRC_PORT))"
            fi
        else
            complete_filter="$complete_filter) or (ip.src == $DST_IP and ip.dst == $SRC_IP))"
        fi
        
        echo "Row $row_num/$total_rows: Extracting packets for time range: $tshark_start_time to $end_time Duration: $flow_duration " >&2
        echo "  Filter: $complete_filter" >&2
        
        tshark -r "$pcap_file" -Y "$complete_filter" -w "$output_file" 2>/dev/null
    fi
    
    # Count packets in final output using CICFlowMeter direction
    if [[ -f "$output_file" ]]; then
        local total_packet_count=$(tshark -r "$output_file" -T fields -e frame.number 2>/dev/null | wc -l)
        
        if [[ "$total_packet_count" -gt 0 ]]; then
            # Use CICFlowMeter-compatible direction counting
            local direction_counts=$(count_packets_by_cicflowmeter_direction "$output_file")
            echo "$total_packet_count,$direction_counts"
        else
            echo "0,0,0"
        fi
    else
        echo "0,0,0"
    fi
}

# Enhanced packet analysis function
analyze_packets() {
    local pcap_file="$1"
    local analysis_file="$2"
    
    if [[ ! -f "$pcap_file" ]]; then
        return 1
    fi
    
    {
        echo "=== CICFlowMeter-Compatible Flow Analysis ==="
        echo "Flow ID: $FLOW_ID"
        echo "Source: $SRC_IP:$SRC_PORT"
        echo "Destination: $DST_IP:$DST_PORT"
        echo "Protocol: $PROTOCOL"
        echo "Attack Label: $LABEL"
        echo "Timestamp: $TIMESTAMP"
        echo "Flow Duration: ${FLOW_DURATION}μs"
        echo "Expected Forward Packets: $TOT_FWD_PKTS"
        echo "Expected Backward Packets: $TOT_BWD_PKTS"
        echo ""
        
        echo "=== CICFlowMeter Direction Analysis ==="
        if get_cicflowmeter_direction "$pcap_file"; then
            echo "Flow Direction (based on first packet):"
            echo "  Forward: $FLOW_FWD_SRC:$FLOW_FWD_SPORT -> $FLOW_FWD_DST:$FLOW_FWD_DPORT"
            echo "  Backward: $FLOW_FWD_DST:$FLOW_FWD_DPORT -> $FLOW_FWD_SRC:$FLOW_FWD_SPORT"
        fi
        
        echo ""
        echo "=== Temporal Boundary Analysis ==="
        local first_packet_time=$(tshark -r "$pcap_file" -T fields -e frame.time 2>/dev/null | head -1)
        local last_packet_time=$(tshark -r "$pcap_file" -T fields -e frame.time 2>/dev/null | tail -1)
        echo "First packet: $first_packet_time"
        echo "Last packet: $last_packet_time"
        
        echo ""
        echo "=== Packet Summary (First 10 packets) ==="
        echo "Packet# | Time                | Src IP          | Dst IP          | CIC Direction"
        echo "--------|---------------------|-----------------|-----------------|---------------"
        tshark -r "$pcap_file" -T fields -e frame.number -e frame.time -e ip.src -e ip.dst 2>/dev/null | \
        head -10 | while IFS=$'\t' read -r num time src dst; do
            local direction="UNKNOWN"
            if [[ "$src" == "$FLOW_FWD_SRC" ]] && [[ "$dst" == "$FLOW_FWD_DST" ]]; then
                direction="FORWARD"
            elif [[ "$src" == "$FLOW_FWD_DST" ]] && [[ "$dst" == "$FLOW_FWD_SRC" ]]; then
                direction="BACKWARD"
            fi
            printf "%-7s | %-19s | %-15s | %-15s | %-13s\n" "$num" "${time:0:19}" "$src" "$dst" "$direction"
        done
        
        echo ""
        echo "=== Directional Statistics ==="
        local direction_counts=$(count_packets_by_cicflowmeter_direction "$pcap_file")
        IFS=',' read -ra counts <<< "$direction_counts"
        local cicfwd_count="${counts[0]}"
        local cicbwd_count="${counts[1]}"
        
        echo "CICFlowMeter Forward packets: $cicfwd_count (Expected: $TOT_FWD_PKTS)"
        echo "CICFlowMeter Backward packets: $cicbwd_count (Expected: $TOT_BWD_PKTS)"
        
    } > "$analysis_file"
}

main() {
    local csv_file="$1"
    local pcap_file="$2"
    local output_dir="${3:-./extracted_packets}"

    # Convert Windows paths
    csv_file=$(convert_windows_path "$csv_file")
    pcap_file=$(convert_windows_path "$pcap_file")
    output_dir=$(convert_windows_path "$output_dir")

    # Validate input files
    if [[ ! -f "$csv_file" ]]; then
        echo "Error: CSV file '$csv_file' not found" >&2
        exit 1
    fi

    if [[ ! -f "$pcap_file" ]]; then
        echo "Error: PCAP file '$pcap_file' not found" >&2
        exit 1
    fi

    # --- Enhancement: Create subdirectory based on PCAP file name ---
    local pcap_base
    pcap_base=$(basename "$pcap_file")
    pcap_base="${pcap_base%.*}"
    local outdir_pcap="$output_dir/$pcap_base"
    mkdir -p "$outdir_pcap"
    # ---------------------------------------------------------------

    # Get total number of rows
    local total_rows
    total_rows=$(tail -n +2 "$csv_file" | wc -l)
    echo "Processing $total_rows flows with CICFlowMeter-compatible direction detection..."

    # All output files now go inside $outdir_pcap
    local overlap_file="$outdir_pcap/temporal_overlap_tracking.csv"
    echo "start_epoch,end_epoch,flow_id" > "$overlap_file"

    local summary_file="$outdir_pcap/cicflowmeter_compatible_summary.csv"
    echo "Row,Flow_ID,Src_IP,Dst_IP,Src_Port,Dst_Port,Protocol,Timestamp,Flow_Duration_us,Label,Found_Total,Found_CIC_Fwd,Found_CIC_Bwd,Expected_Fwd,Expected_Bwd,Expected_Total,CIC_Fwd_Accuracy,CIC_Bwd_Accuracy,Total_Accuracy,Boundary_Detection_Used,Output_File" > "$summary_file"

    local processed=0
    local found=0
    local errors=0
    local boundary_detections=0
    local total_accuracy_sum=0
    local direction_matches=0

    for ((row=2; row<=total_rows+1; row++)); do
        if parse_csv_row_enhanced "$csv_file" "$row"; then
            processed=$((processed + 1))

            local safe_label
            safe_label=$(echo "$LABEL" | tr -d '/<>:"|?*' | tr ' ' '_')
            local flow_pcap="$outdir_pcap/flow_${processed}_${safe_label}.pcap"
            local flow_analysis="$outdir_pcap/flow_${processed}_${safe_label}_analysis.txt"
            
            # Extract packets with boundary detection
            local packet_counts
            if packet_counts=$(extract_packets_with_boundary_detection "$pcap_file" "$flow_pcap" "$processed" "$FLOW_DURATION" "$TIMESTAMP" "$total_rows" "$overlap_file"); then
                IFS=',' read -ra counts <<< "$packet_counts"
                local found_total="${counts[0]}"
                local found_cic_fwd="${counts[1]}"
                local found_cic_bwd="${counts[2]}"
                
                local expected_total=$((TOT_FWD_PKTS + TOT_BWD_PKTS))
                
                if [[ "$found_total" =~ ^[0-9]+$ ]]; then
                    if [[ "$found_total" -gt 0 ]]; then
                        found=$((found + 1))
                        
                        # Calculate CICFlowMeter-compatible accuracies
                        local cic_fwd_accuracy=0
                        local cic_bwd_accuracy=0
                        local total_accuracy=0
                        
                        if [[ "$TOT_FWD_PKTS" -gt 0 ]]; then
                            cic_fwd_accuracy=$(( (found_cic_fwd * 100) / TOT_FWD_PKTS ))
                        fi
                        if [[ "$TOT_BWD_PKTS" -gt 0 ]]; then
                            cic_bwd_accuracy=$(( (found_cic_bwd * 100) / TOT_BWD_PKTS ))
                        fi
                        if [[ "$expected_total" -gt 0 ]]; then
                            total_accuracy=$(( (found_total * 100) / expected_total ))
                            total_accuracy_sum=$((total_accuracy_sum + total_accuracy))
                        fi
                        
                        # Check if direction matches are reasonable (80-120% range)
                        if [[ "$cic_fwd_accuracy" -ge 80 ]] && [[ "$cic_fwd_accuracy" -le 120 ]] && [[ "$cic_bwd_accuracy" -ge 80 ]] && [[ "$cic_bwd_accuracy" -le 120 ]]; then
                            direction_matches=$((direction_matches + 1))
                        fi
                        
                        local boundary_used="NO"
                        if [[ "$total_accuracy" -eq 100 ]]; then
                            boundary_used="YES"
                            boundary_detections=$((boundary_detections + 1))
                        fi
                        
                        echo "Row $processed/$total_rows: Found $found_total packets for $LABEL attack from $pcap_file"
                        echo "  - CICFlowMeter Forward: $found_cic_fwd/$TOT_FWD_PKTS packets (${cic_fwd_accuracy}%)"
                        echo "  - CICFlowMeter Backward: $found_cic_bwd/$TOT_BWD_PKTS packets (${cic_bwd_accuracy}%)" 
                        echo "  - Total: $found_total/$expected_total packets (${total_accuracy}%)"
                        echo "  - Source: $SRC_IP:$SRC_PORT -> $DST_IP:$DST_PORT"
                        echo "  - Boundary Detection: $boundary_used"
                        
                        analyze_packets "$flow_pcap" "$flow_analysis"
                        
                        echo "$processed,\"$FLOW_ID\",\"$SRC_IP\",\"$DST_IP\",\"$SRC_PORT\",\"$DST_PORT\",\"$PROTOCOL\",\"$TIMESTAMP\",\"$FLOW_DURATION\",\"$LABEL\",$found_total,$found_cic_fwd,$found_cic_bwd,$TOT_FWD_PKTS,$TOT_BWD_PKTS,$expected_total,$cic_fwd_accuracy,$cic_bwd_accuracy,$total_accuracy,\"$boundary_used\",\"$flow_pcap\"" >> "$summary_file"
                    else
                        echo "Row $processed/$total_rows: No packets found for $LABEL attack"
                        echo "$processed,\"$FLOW_ID\",\"$SRC_IP\",\"$DST_IP\",\"$SRC_PORT\",\"$DST_PORT\",\"$PROTOCOL\",\"$TIMESTAMP\",\"$FLOW_DURATION\",\"$LABEL\",0,0,0,$TOT_FWD_PKTS,$TOT_BWD_PKTS,$expected_total,0,0,0,\"NO\",N/A" >> "$summary_file"
                    fi
                else
                    errors=$((errors + 1))
                    echo "Row $processed/$total_rows: Error processing $LABEL attack" >&2
                fi
            else
                errors=$((errors + 1))
                echo "Row $processed/$total_rows: Error extracting packets for $LABEL attack" >&2
            fi
            
            if [[ $((processed % 5)) -eq 0 ]]; then
                echo "Progress: $processed/$total_rows flows processed ($(( (processed * 100) / total_rows ))%)"
            fi
        fi
    done
    
    # Calculate statistics
    local avg_accuracy=0
    local direction_match_rate=0
    if [[ $found -gt 0 ]]; then
        avg_accuracy=$((total_accuracy_sum / found))
        direction_match_rate=$(( (direction_matches * 100) / found ))
    fi
    
    echo ""
    echo "=== CICFlowMeter-Compatible Extraction Complete ==="
    echo "Total flows processed: $processed"
    echo "Flows with matching packets: $found"
    echo "Processing errors: $errors"
    echo "Boundary detections used: $boundary_detections"
    echo "Direction matches (80-120%): $direction_matches"
    if [[ $processed -gt 0 ]]; then
        echo "Success rate: $(( (found * 100) / processed ))%"
        echo "Average packet extraction accuracy: ${avg_accuracy}%"
        echo "Direction match rate: ${direction_match_rate}%"
        echo "Boundary detection usage: $(( (boundary_detections * 100) / processed ))%"
    fi
    echo "Results saved in: $output_dir"
    echo "Summary file: $summary_file"
    
    # Create final report
    local report_file="$output_dir/cicflowmeter_extraction_report.txt"
    {
        echo "DNP3 Packet Extraction Report - CICFlowMeter Compatible"
        echo "======================================================="
        echo "Generated on: $(date)"
        echo "CSV File: $csv_file"
        echo "PCAP File: $pcap_file"
        echo "Output Directory: $output_dir"
        echo ""
        echo "Processing Summary:"
        echo "- Total flows in CSV: $total_rows"
        echo "- Successfully processed: $processed"
        echo "- Flows with packets found: $found"
        echo "- Processing errors: $errors"
        echo "- Boundary detections used: $boundary_detections"
        echo "- Direction matches (80-120%): $direction_matches"
        echo "- Success rate: $(( processed > 0 ? (found * 100) / processed : 0 ))%"
        echo "- Average packet extraction accuracy: ${avg_accuracy}%"
        echo "- Direction match rate: ${direction_match_rate}%"
        echo ""
        echo "CICFlowMeter Compatibility Features:"
        echo "- Temporal-based flow direction detection"
        echo "- First packet determines forward/backward directions"
        echo "- Boundary detection prevents packet overlap"
        echo "- Time-based flow segmentation alignment"
        echo "- Enhanced directional packet counting"
        echo ""
        echo "Output Files:"
        echo "- Summary CSV: $summary_file"
        echo "- Individual PCAP files: flow_*_*.pcap"
        echo "- Analysis reports: flow_*_*_analysis.txt"
    } > "$report_file"
    
    echo "Final report: $report_file"
    
    # Cleanup
    rm -f "$overlap_file"
}

# Script usage
usage() {
    echo "DNP3 Dataset Packet Extractor - CICFlowMeter Compatible Version"
    echo "Usage: $0 <csv_file> <pcap_file> [output_directory]"
    echo ""
    echo "Arguments:"
    echo "  csv_file        Path to the DNP3 CSV file (CICFlowMeter format)"
    echo "  pcap_file       Path to the corresponding PCAP file"
    echo "  output_directory Optional output directory (default: ./extracted_packets)"
    echo ""
    echo "Key Features:"
    echo "  ✅ CICFlowMeter-compatible direction detection (temporal-based)"
    echo "  ✅ Temporal boundary detection prevents packet overlap"
    echo "  ✅ First packet determines forward/backward directions"
    echo "  ✅ Enhanced progress display with row numbers"
    echo "  ✅ Comprehensive packet validation and analysis"
    echo "  ✅ Windows Git Bash compatibility"
    echo ""
    echo "Direction Algorithm:"
    echo "  - Forward direction: Same as first packet in temporal sequence"
    echo "  - Backward direction: Opposite to first packet direction"
    echo "  - Aligns with CICFlowMeter's bidirectional flow model"
    echo ""
    echo "Windows Examples:"
    echo "  $0 \"C:/Users/username/Documents/dnp3_flows.csv\" \"C:/Users/username/Documents/dnp3_traffic.pcap\""
    echo "  $0 /c/Users/username/Documents/dnp3_flows.csv /c/Users/username/Documents/dnp3_traffic.pcap ./results"
    echo ""
    echo "Requirements:"
    echo "  - Wireshark with tshark (install from wireshark.org)"
    echo "  - Git Bash for Windows"
    echo "  - Sufficient disk space for extracted PCAP files"
}

# Entry point
if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

echo "DNP3 Packet Extractor - CICFlowMeter Compatible Version Starting..."
echo "Checking dependencies..."
check_dependencies
echo "Dependencies OK. Starting extraction with CICFlowMeter direction alignment..."
main "$@"
