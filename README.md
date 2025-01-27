# FACT-Net  
**FACT-Net: A Two-Stage Fusion-CNN-Transformer Framework for ABP Signal Reconstruction in Cross-Platform Multi-Patient IoT Healthcare Systems**

## Multimodal Physiological Signal Acquisition System  
The multimodal physiological signal acquisition system developed in this study integrates advanced hardware modules to ensure precise and efficient data acquisition for ABP reconstruction tasks.  

### System Components:  
- **(a)** Sensor unit, signal processing circuitry, microcontroller unit (MCU), data processing module, and charging unit.  
- **(b)** A simulation diagram illustrating the functional design of the acquisition system.  
- **(c)** Front view of the acquisition device, highlighting compactness and ergonomic design.  
- **(d)** Rear view showcasing the modularity of system components.  
- **(e)** PPG sensor for high-accuracy physiological signal collection.  
- **(f)** 3D-printed casing integrated with a lithium battery, designed for enhanced portability and durability.  

![System Overview](https://github.com/liuyisi123/FACT-Net/blob/main/Hardware.png)  

## FACT-Net Architecture  
FACT-Net is a two-stage hybrid model tailored for ABP waveform reconstruction:  

### Stage I: Parallel Cross-Hybrid Architecture  
- Combines **Multi-Scale CNN Blocks** and **Mix-T Blocks** to extract multimodal features and provide constraint information.  

### Stage II: Serial Hybrid CNN-Transformer Structure  
- Refines feature representation and achieves high-fidelity ABP waveform reconstruction.  

### Key Architectural Details:  
- **Multi-Scale CNN Blocks**: Designed to capture hierarchical features across varying temporal scales.  
- **Mix-T Blocks**: Facilitate multimodal feature fusion for enhanced signal integration.  

![FACT-Net Architecture](https://github.com/liuyisi123/FACT-Net/blob/main/FACT-Net.png)  

## Cross-Platform Multi-Patient IoT Framework (CPMP-IoT)  
The CPMP-IoT framework extends FACT-Net's capabilities to real-world healthcare scenarios, enabling cross-platform, multi-patient health management:  

- **(a)** **Individual Monitoring APP**: Supports offline inference for personalized health monitoring.  
- **(b)** **Host Computer Integration**: FACT-Net-enabled devices can connect via LAN, allowing multiple IoT devices to access health reports and data through a web-based interface.  
- **(c)** **Ward-Level Multi-Patient Monitoring**: Facilitates simultaneous monitoring and management of multiple patients in healthcare wards, ensuring scalability and reliability.  

![CPMP-IoT Framework](https://github.com/liuyisi123/FACT-Net/blob/main/CPMP-IoT.png)  

