# CoastVision - Detailed Team Roles & Contributions

## Team Structure & Impact Hierarchy
1. **Harshal** (Team Lead & Core Architect) - *Highest Impact*
2. **Hardik** (Lead Frontend Engineer) - *High Impact*
3. **Sara** (Research, Analytics & Deployment) - *Significant Impact*
4. **Komal** (Data Curation & API Integration) - *Supporting Impact*

## Detailed Technical Contributions

### 1. Harshal - Team Lead & Core AI/Backend Architect (Highest Impact)
*Harshal spearheaded the entire project, taking sole ownership of the core AI inference engine, streaming pipeline, and backend architecture.*
- **System Architecture:** Designed the multi-threaded Flask backend running on a Waitress production WSGI server. Implemented a concurrency model where each camera feed operates within its own isolated daemon worker thread, preventing system-wide blocks during intensive processing.
- **AI & Model Pipeline:** Built a dynamic dual-model YOLO architecture: a custom `best.pt` model for drowning/swimming detection that seamlessly falls back to a COCO model for person tracking. Configured the PyTorch environment for the RTX 3050 GPU by enabling TF32 Tensor Cores, FP16 half-precision, and cuDNN auto-tuning, successfully reducing inference latency to 15-25ms per frame.
- **Advanced HLS Video Streaming:** Engineered a hardware-accelerated HLS (HTTP Live Streaming) pipeline to eliminate high bandwidth usage. Programmed the backend to pipe raw BGR frames directly into an FFmpeg subprocess, utilizing the `h264_nvenc` NVIDIA hardware encoder to compress and serve video segments with near-zero latency.
- **Project Lead:** Directed all architectural decisions, managed the integration between the machine learning models and the web server, and delivered a production-ready codebase.

### 2. Hardik - Lead Frontend Engineer (High Impact)
*Hardik translated Harshal's high-speed backend data into an interactive, high-performance web dashboard.*
- **Dashboard UI/UX:** Built the entire frontend architecture using React 18, Vite, and Material-UI (MUI). Implemented state management to handle dynamic zone rendering, allowing the live monitoring grid to automatically adapt when new cameras are added.
- **Fault-Tolerant Video Player:** Engineered the `ZoneStreamView` component to handle unstable beach networks. Implemented a programmatic fallback chain that utilizes `hls.js` for smooth hardware-decoded video, but monitors network health to instantly downgrade to MJPEG or frame polling if packet loss is detected.

- **Browser API Integrations:** Developed custom React hooks (`useEmergencyVoiceAlert`) to tap into the browser's native Web Speech API and AudioContext API. This implementation synthesizes voice announcements and alarm beeps natively on the client side the exact moment a high-confidence alert is received.

### 3. Sara - Research, Analytics & Deployment Specialist (Significant Impact)
*Sara led the scientific research, mathematically tuned the AI models, and handled deployment architecture.*
- **Research & System Analytics:** Spearheaded theoretical research into computer vision for coastal safety. Analyzed the severe bandwidth limitations of existing MJPEG systems and mathematically validated Harshal's HLS pipeline by proving it reduced network load from 36 MB/s down to 3 MB/s.
- **Model Evaluation & Scientific Tuning:** Executed rigorous validation scripts (`evaluate_model.py`) to extract precision, recall, and mAP metrics. Used this data to systematically configure the backend environment variables, setting the emergency threshold to 0.55 to perfectly balance high recall without causing alarm fatigue.
- **DevOps & System Testing:** Created the deployment infrastructure by writing automated Windows PowerShell scripts that establish virtual environments and install dependencies with a single click. Conducted stress testing on backend concurrency by simulating simultaneous high-resolution video streams.


### 4. Komal - Data Curation & Alert Integration (Supporting Impact)
*Komal provided essential support by preparing the AI training data and developing the external messaging and mobile routing logic.*
- **Dataset Management:** Sourced the core dataset from Roboflow Universe. Managed the data pipeline by curating, cleaning, and normalizing thousands of annotated images, ensuring the bounding-box labels were perfectly structured for the YOLOv8 training process.
- **Telegram Bot & Routing Logic:** Engineered the integration between the Flask backend and the Telegram messaging API. Implemented stateful routing logic using `telegram_users.json` to map specific lifeguard IDs to specific camera zones, ensuring targeted alert delivery.

- **Visual Analytics UI:** Developed the Analytics tab using `react-chartjs-2`. Implemented data polling hooks to fetch backend timelines and rendered the `PersonCountTimeline` component, configuring gradient fills, synchronized tooltips, and dynamic Y-axis scaling to visualize crowd density.