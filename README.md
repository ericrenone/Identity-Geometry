# FPGA Novelty Detection Accelerator
**Hardware-Software-Hyrbrid-Accelerated Leaky Integrator Neuron for Deterministic Anomaly Detection**

---

## **1. Overview**
A **minimalist, hardware-accelerated leaky integrator neuron** implemented on the **Gowin Tang Nano 9K FPGA**. Designed for **real-time novelty detection** with **deterministic timing** and **ultra-low resource usage**.

---

## **2. Technical Architecture**
### **2.1 Core Logic**
- **State Transition Function**:
  `f(ϵ, x) = (ϵ >> 1) + x`
  - `ϵ`: 16-bit energy accumulator.
  - `x`: 8-bit UART input.
  - `>> 1`: Bit-shift decay (division-by-2).

### **2.2 Hardware-Software Co-Design**
- **Digital Twin**: Bit-exact Python mirror for verification.
- **UART Echo**: Real-time benchmarking.
- **LED Visualization**: Novelty event feedback.

---

## **3. Performance Metrics**
| Metric                | FPGA (Hardware)       | CPU (Software)       |
|-----------------------|-----------------------|-----------------------|
| **Compute Latency**    | 37.037 ns              | N/A                   |
| **Transport Latency**  | ~86.8 µs (UART)       | N/A                   |
| **Jitter**            | < 0.5 µs              | High (OS-dependent)   |
| **Numeric Format**    | Q16.0 Fixed-Point     | 64-bit Float/Int      |

### **Key Insights**
- **I/O-Bound System**: Transport latency dominates compute time.
- **Deterministic Timing**: FPGA achieves **< 0.5 µs jitter**.
- **Energy Efficiency**: Multiplier-less design minimizes power.

---

## **4. Uniqueness**
1. **Memory Wall Justification**: Exposes **1,000x latency penalty** of serial transport.
2. **Q-Format Stability**: 16-bit fixed-point suffices for bio-inspired dynamics.
3. **Deterministic Timing**: Temporal precision > numerical precision.
4. **Optimal Energy**: Shift-accumulate logic reaches theoretical efficiency.
5. **Formal Veracity**: Lambda Calculus mapping ensures correctness.

---

## **5. File Structure**
| File                | Description                          |
|---------------------|--------------------------------------|
| `Verilog.txt`       | FPGA RTL (Tang Nano 9K).             |
| `Benchmark.py`      | Anomaly injection & latency test.    |
| `FGPA_CPU_Test.py`  | Real-time visualization dashboard.  |

---

## **6. Conclusions**
- **Hardware Superiority**: FPGA provides **deterministic latency**.
- **Neuromorphic Success**: Leaky integrator accurately models "energy" decay.
- **I/O Bottleneck**: UART transport limits performance, not compute.
- **Reproducibility**: Fixed seeds ensure scientific repeatability.

---

## **7. Next Steps**
- **Adaptive Thresholding**: Implement STDP for dynamic learning.
- **Functional HDL Port**: Rewrite in Clash/Bluespec for Lambda Calculus integration.
- **Scalability**: Extend for federated learning (FL).

---
**Note**: This project demonstrates **deterministic, low-power anomaly detection** on **cheap FPGAs**, paving the way for **edge AI** and **large-scale device networks**.
