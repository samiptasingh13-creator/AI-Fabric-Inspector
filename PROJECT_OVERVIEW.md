
# Research Objectives:

The primary goal of this project is to build a practical, real-time automated inspection system for industrial textile manufacturing that runs efficiently on standard Industrial PCs (IPCs).

High-Speed Detection: Maintain frame rates above 80 FPS to match real-time production speeds without introducing latency in conveyor sorting systems.

Micro-Defect Accuracy: Reliably catch small, low-contrast anomalies—such as micro-holes, oil spots, and thick yarns—that traditional automated systems miss or merge into background weave patterns.

Edge-Friendly Efficiency: Keep the model lightweight (under 4 Million parameters) so it can deploy on low-cost hardware without requiring heavy industrial GPUs.

Model Transparency: Integrate visual heatmaps (Grad-CAM) into the operator dashboard so quality control teams can verify that decisions are based on real fabric flaws rather than background noise.

# Methodology:

Our system combines targeted image enhancement, a lightweight feature-preserving network, and automated post-processing:

Image Preprocessing: We apply adaptive Non-Local Means (NLM) filtering alongside local contrast normalization. This smooths out high-frequency warp-and-weft weave patterns while sharpening the boundaries of actual fabric defects.

Architecture Design: The network uses depthwise separable convolutions to cut down parameter count, paired with a Feature Pyramid Network (FPN) to retain fine spatial details across multiple scales.

Loss Functions: Training utilizes Focal Loss to handle severe class imbalance (where defect-free fabric vastly outnumbers flaw regions) and GIoU loss for precise bounding box placement.

Grading & Hardware Loop: Inspection results feed directly into a state machine that assigns Grade A (Pass), Grade B (Minor Flaw / Divert), or Grade C (Critical Defect / Stop Line) flags, triggering physical sorting mechanisms or alerting operators via an interactive Streamlit UI.

# Key Results:

Across 1,200 test images, the system demonstrated strong performance, striking a solid balance between processing speed and accuracy:

Detection Accuracy: Achieved 93.3% mAP@0.5 overall (95.8% on oil stains, 94.0% on thick yarns, and 90.1% on micro-holes).

Inference Speed: Runs at 80.6 FPS with an average inference time of 12.4 ms per image on an Intel i7 IPC setup.

Model Footprint: Compact total parameter count of 3.9M and a overall model size of 15.6 MB—over 10× smaller than baseline models like Faster R-CNN (41.5M parameters) with virtually no loss in practical accuracy.