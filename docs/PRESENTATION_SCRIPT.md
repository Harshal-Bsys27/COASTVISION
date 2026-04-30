# CoastVision - Academic Defense & Presentation Script

**Project Title:** CoastVision - AI-powered coastal surveillance system
**Team Members:** Harshal (Lead), Hardik, Sara, Komal
**Estimated Time:** 12-15 Minutes

---

## 1. Introduction & Core Backend Architecture
**(Speaker: Harshal - Team Lead)**

**Harshal:**
"Good morning, respected panel members and guests. We are Team CoastVision. My name is Harshal, and as the Team Lead and Core Architect, I will be initiating our presentation today. I am joined by my colleagues Hardik, Sara, and Komal.

We built CoastVision to address a severe, life-threatening problem: silent drowning. Human vigilance is prone to fatigue, so we asked ourselves—how can we use computer vision to provide lifeguards with an infallible, real-time second set of eyes?

My responsibility was building the brain of this operation. To process high-definition video in real-time, I engineered a highly concurrent, multi-threaded Flask backend. Inside this backend, I implemented a dynamic, dual-model YOLO architecture. We run a custom YOLOv8 model explicitly trained for drowning detection, which seamlessly falls back to a COCO model for person tracking. However, running deep learning models is computationally expensive. To achieve real-time latency, I configured our PyTorch pipeline to fully utilize our RTX 3050 GPU—enabling TF32 Tensor Cores and FP16 half-precision. This engineering reduced our inference latency down to an incredibly fast 15 to 25 milliseconds per frame.

Finally, sending raw video across a network is notoriously bandwidth-heavy. To solve this, I engineered a hardware-accelerated HLS streaming pipeline. The backend pipes raw video frames directly into an FFmpeg subprocess, utilizing the NVIDIA hardware encoder to compress and stream the video with near-zero latency.

Now, I would like to invite Hardik to explain how he engineered the frontend to consume and display this highly complex data stream."

---

## 2. Frontend UI, React Dashboard & Fallback Streaming
**(Speaker: Hardik)**

**Hardik:**
"Thank you, Harshal, and good morning everyone. As the Lead Frontend Engineer, my primary challenge was designing an interface that could handle Harshal's high-speed video streams without ever freezing or crashing during an emergency.

I architected the entire frontend using React 18, Vite, and Material-UI. The focal point of the dashboard is our multi-zone Live Monitoring grid. We knew that network connections at beaches are notoriously unstable. Therefore, I engineered the video player with a programmatic fallback chain. 

Here is how it works: the player attempts to use the hardware-decoded HLS stream first. If my code detects network packet loss or instability, it instantly triggers a state change to downgrade to MJPEG streaming, and if the network drops further, it falls back to single-frame polling. This ensures the lifeguard's visual feed never freezes.

Furthermore, a visual alert isn't enough if a lifeguard is looking away. To solve this, I wrote custom React hooks that tap directly into the browser's native Web Speech API and AudioContext API. The exact millisecond an emergency is detected, the dashboard synthesizes a vocal announcement and an audible alarm natively on the client's machine.

Next, I'll hand the floor over to Sara, who will explain the rigorous research and methodology ensuring our system is scientifically sound."

---

## 3. Research, Model Tuning & DevOps
**(Speaker: Sara)**

**Sara:**
"Thank you, Hardik. A safety system is only as valuable as its reliability. My role as the Research, Deployment, and Testing Specialist was to ensure our architectural choices were mathematically sound and that our AI was highly accurate.

Early in the project, I spearheaded the theoretical research into existing surveillance systems. A major part of my role was validating Harshal's HLS streaming architecture. By running mathematical bandwidth analyses, I proved that our HLS implementation successfully reduced the network load from a massive 36 megabytes per second down to just 3 megabytes per second, validating our system for low-bandwidth coastal areas.

Once the AI models were trained, I was responsible for tuning them. I executed rigorous validation scripts to extract precision, recall, and mAP metrics. Using this empirical data, I scientifically fine-tuned our backend environment variables. I configured the emergency threshold to 0.55—a calculated sweet spot that perfectly balances high recall without causing false-positive alarm fatigue for the lifeguards.

Finally, I designed our deployment automation, writing PowerShell scripts that establish our virtual environments instantly, and conducted severe stress testing to ensure the backend concurrency could handle multiple high-resolution streams simultaneously.

I will now pass the presentation to Komal to discuss our data engineering, visual analytics, and external routing."

---

## 4. Data Curation, Visual Analytics & Alert Routing
**(Speaker: Komal)**

**Komal:**
"Thanks, Sara. Good morning everyone. My role encompassed data engineering, frontend data visualization, and external alert routing.

First, for an AI to be highly accurate, it requires impeccable training data. I managed our entire dataset pipeline by sourcing the core dataset from Roboflow Universe. I curated, cleaned, and normalized thousands of images, ensuring the bounding-box labels were perfectly structured to feed into the YOLOv8 training process.

Secondly, raw data needs to be readable. I took ownership of the Visual Analytics UI on the frontend dashboard. Utilizing `react-chartjs-2`, I implemented data polling hooks to fetch backend timelines and engineered the `PersonCountTimeline` component. I configured gradient fills, synchronized tooltips, and dynamic Y-axis scaling to effectively visualize crowd density and help lifeguards predict busy periods.

Finally, I engineered our external alert integration using the Telegram Bot API. When the backend flags a critical event, my routing logic triggers. By implementing stateful routing with a persistent JSON file, the system accurately maps specific camera zones to specific lifeguards, pushing instant text alerts and snapshots directly to their phones while they are away from the dashboard.

I'll now pass it back to Harshal to conclude our defense."

---

## 5. Conclusion
**(Speaker: Harshal - Team Lead)**

**Harshal:**
"Thank you, Komal. 

In conclusion, CoastVision is not just a prototype; it is a highly advanced, production-ready AI solution. By combining extreme GPU-accelerated deep learning optimizations, robust hardware video encoding, a fault-tolerant React dashboard, rigorous scientific tuning, and mobile alert integration, we have created an enterprise-grade tool that can genuinely assist safety teams in saving lives.

We would like to sincerely thank the panel for your time and attention today. We are now happy to provide a live demonstration and answer any questions you may have."
