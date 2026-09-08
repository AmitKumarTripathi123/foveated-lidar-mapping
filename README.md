# 🚗 SIH 2026: Adaptive Variable-Resolution 2.5D LiDAR Mapping Platform
### Dynamic Environment Perception for Autonomous Navigation

---

## 🌐 🚀 LIVE PRODUCTION DEPLOYMENT (FOR SIH PRESENTATION)

> ### **🔗 Live Cloud Dashboard (Vercel Global CDN):**  
> # **👉 [https://frontend-henna-chi-pu2t4bz6wk.vercel.app](https://frontend-henna-chi-pu2t4bz6wk.vercel.app) 👈**
> *(Alternate Production Alias: [https://frontend-fbtrdw9iu-ayush-misra.vercel.app](https://frontend-fbtrdw9iu-ayush-misra.vercel.app))*

| Component | Production URL | Status | Protocol |
| :--- | :--- | :---: | :---: |
| **Frontend Dashboard** | `https://frontend-henna-chi-pu2t4bz6wk.vercel.app` | `LIVE (200 OK)` | HTTPS (Vercel CDN) |
| **Backend REST Health** | `https://arena-exorcism-unsettled.ngrok-free.dev/api/v1/health` | `HEALTHY` | HTTPS (Express API) |
| **Real-Time Stream** | `wss://arena-exorcism-unsettled.ngrok-free.dev/ws/stream` | `ACTIVE (10 Hz)` | Secure WebSocket (WSS) |

[![Vercel Deployment](https://img.shields.io/badge/Vercel-LIVE%20PRODUCTION-success?style=for-the-badge&logo=vercel)](https://frontend-henna-chi-pu2t4bz6wk.vercel.app)
[![Node.js](https://img.shields.io/badge/Node.js-v20+-green.svg?style=for-the-badge&logo=node.js&logoColor=white)](https://nodejs.org)
[![Next.js](https://img.shields.io/badge/Next.js-14.1-black.svg?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL%202.5D-blue.svg?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4-blue.svg?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)

---

## 📌 Problem Statement & Core Innovation

Traditional autonomous vehicle perception discretizes the surrounding environment into a **uniform fine-resolution grid** (e.g. $5\text{ cm}$ over $100\text{ m}$ radius). This causes catastrophic memory explosions:
- **Uniform Grid ($5\text{ cm}$ across $100\text{ m}$):** Requires **$12,566,370\text{ cells}$** ($\approx 785.4\text{ MB}$ memory footprint), crashing real-time embedded compute boards.
- **Our Innovation (Foveated 2.5D Elevation Grid):** Employs biological foveation with 3 concentric spatial resolution zones, preserving millimetric precision for immediate obstacle/curb avoidance while coarsening the far range:
  - **Zone 0 ($0\text{--}10\text{ m}$):** **$5\text{ cm}$ ($0.05\text{ m}$)** spatial resolution (Near-field ego vehicle corridor & curb detection).
  - **Zone 1 ($10\text{--}50\text{ m}$):** **$25\text{ cm}$ ($0.25\text{ m}$)** spatial resolution (Mid-field dynamic vehicles & pedestrians).
  - **Zone 2 ($50\text{--}100\text{ m}$):** **$50\text{ cm}$ ($0.50\text{ m}$)** spatial resolution (Far-field macro road corridor & boundaries).
- **Result:** **$340,549\text{ cells}$** total address space $\rightarrow$ **$97.29\%$ memory reduction** with $12.1\text{--}13.8\text{ ms}$ grid latency!

---

## 📊 Scientific & Benchmark Invariants (Audited Baseline)

| Metric | Uniform Baseline ($0.05\text{m}$) | Our Foveated 2.5D Grid | Efficiency Gain / Status |
| :--- | :---: | :---: | :---: |
| **Theoretical Grid Capacity** | $12,566,370\text{ cells}$ | **$340,549\text{ cells}$** | **$-97.29\%$ Memory Reduction** |
| **Reference Benchmark (Frame 0)** | $45,820\text{ cells}$ | **$9,169\text{ cells}$** | **$-80.0\%$ Occupied Cell Reduction** |
| **Grid Processing Latency** | $55.6\text{ ms}$ | **$12.1\text{--}13.8\text{ ms}$** | **$4.6\times\text{ Faster Processing}$** |
| **AI Perception Inference (RTX 3070)** | $18.2\text{ ms}$ | $18.2\text{ ms}$ | SPVCNN Backbone Reference |
| **Derived Pipeline Latency** | $73.8\text{ ms}$ | **$30.3\text{--}32.0\text{ ms}$** | Real-Time Realized ($>30\text{ Hz}$) |
| **Pipeline Throughput** | $13.5\text{ Hz}$ | **$31.3\text{--}33.0\text{ Hz}$** | Exceeds 10 Hz LiDAR sensor spin |
| **Estimated Buffer Footprint** | $785.4\text{ MB}$ | **$21.8\text{ MB}$** | $64\text{ B/cell struct model}$ |

---

## 🎬 Quick SIH Demo Presentation Flow (2 Minutes)

When presenting to SIH judges, follow this concise 4-step sequence:

1. **Launch Live URL:** Open **[https://frontend-henna-chi-pu2t4bz6wk.vercel.app](https://frontend-henna-chi-pu2t4bz6wk.vercel.app)** on screen.
2. **Demonstrate Contiguous 2.5D Map:**
   - Show the continuous road surface spanning $-75\text{m}$ to $+75\text{m}$ (Class 0: Drivable Terrain, Green `#22C55E`).
   - Show curbs (15cm elevated, Amber `#CA8A04`), sidewalks, and buildings with zero artificial voids.
3. **Showcase Variable-Resolution Foveation:**
   - Point to the concentric boundary rings at $10\text{m}$, $50\text{m}$, and $100\text{m}$.
   - Demonstrate the visible density transition: $5\text{cm}$ near $\rightarrow$ $25\text{cm}$ mid $\rightarrow$ $50\text{cm}$ far.
4. **Interactive Camera Navigation:**
   - Use the **Camera Distance Slider** ($15\text{m}$ to $220\text{m}$) to smoothly zoom into the ego vehicle or out to 100m.
   - Click corridor focus buttons: **`Ego`** ($Y=0\text{m}$), **`Center`** ($Y=15\text{m}$), and **`Ahead`** ($Y=32\text{m}$).
   - Switch Color Mode to **`Elevation`** to highlight ground height gradients from Blue (low) to Red (high).
5. **Open Benchmark Modal:**
   - Click **`BENCHMARK COMPARISON`** in the top header to display the analytical breakdown showing the **$97.29\%$ address reduction**.

---

## 💻 Local Development Setup (For Offline Presentations)

If you wish to run the entire stack locally on your presentation laptop:

### Terminal 1: Backend Perception Server
```powershell
cd backend
npm install
npm run dev
```
- REST Health: `http://localhost:8000/api/v1/health`
- WebSocket: `ws://localhost:8000/ws/stream`

### Terminal 2: Frontend 3D Dashboard
```powershell
cd frontend
npm install
npm run dev
```
- Dashboard URL: `http://localhost:3000`

---

## 👥 Core Engineering Team

- **AYUSH (Lead Full-Stack & Systems Integration Engineer)**: Node.js/Express Backend, WebSocket streaming engine, Next.js / Three.js 2.5D WebGL visualizer, camera orbit/zoom systems, production deployment.
- **Amit**: Raw LiDAR dataset ingestion, coordinate normalization (`.bin` / `.pcd`), sequence packaging.
- **Atul**: SPVCNN AI perception pipeline, 3D semantic segmentation, bounding box inference.
- **Ankur**: Variable-resolution foveated grid mathematical modeling and projection algorithms.
