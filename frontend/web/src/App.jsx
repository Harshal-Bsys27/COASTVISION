import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import Hls from "hls.js";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  TimeScale,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import {
  AppBar,
  Box,
  Button,
  Chip,
  Dialog,
  Divider,
  IconButton,
  Stack,
  Tab,
  Tabs,
  Toolbar,
  Typography,
  Card,
  CardHeader,
  Avatar,
  Badge,
  Tooltip,
  LinearProgress,
  TextField,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import RefreshIcon from "@mui/icons-material/Refresh";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import WarningIcon from "@mui/icons-material/Warning";
import VideocamIcon from "@mui/icons-material/Videocam";
import NotificationsActiveIcon from "@mui/icons-material/NotificationsActive";
import DashboardIcon from "@mui/icons-material/Dashboard";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import HistoryIcon from "@mui/icons-material/History";
import SettingsIcon from "@mui/icons-material/Settings";
import WavesIcon from "@mui/icons-material/Waves";
import FiberManualRecordIcon from "@mui/icons-material/FiberManualRecord";
import PersonIcon from "@mui/icons-material/Person";
import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import VolumeOffIcon from "@mui/icons-material/VolumeOff";
import DownloadIcon from "@mui/icons-material/Download";
import SpeedIcon from "@mui/icons-material/Speed";
import CenterFocusStrongIcon from "@mui/icons-material/CenterFocusStrong";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import ZoomInIcon from "@mui/icons-material/ZoomIn";
import ZoomOutIcon from "@mui/icons-material/ZoomOut";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import VideoLibraryIcon from "@mui/icons-material/VideoLibrary";
import DeleteIcon from "@mui/icons-material/Delete";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import EditIcon from "@mui/icons-material/Edit";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import SecurityIcon from "@mui/icons-material/Security";
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";
import LifeguardAccountsPanel from "./components/LifeguardAccountsPanel.jsx";
import ResponseTimeAnalytics from "./components/ResponseTimeAnalytics.jsx";
import CrowdDensityAnalytics from "./components/CrowdDensityAnalytics.jsx";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  TimeScale,
  Filler
);

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

// Tune card + fallback rates
const GRID_W = 640;
const FALLBACK_FRAME_MS = 180;

// FIX: this was missing but used in the modal polling effect -> caused blank screen on click
const MODAL_REFRESH_MS = 140;

function usePollJson(url, ms, enabled, initial) {
  const [data, setData] = useState(initial);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(url, { cache: "no-store" });
        const j = await r.json();
        if (alive) setData(j);
      } catch {}
    };
    tick();
    const t = setInterval(tick, ms);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [url, ms, enabled]);
  return data;
}

function usePollJsonWithOk(url, ms, enabled, initial) {
  const [state, setState] = useState({ data: initial, ok: null });
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(url, { cache: "no-store" });
        const ok = !!r.ok;
        const j = await r.json();
        if (alive) setState({ data: j, ok });
      } catch {
        if (alive) setState((s) => ({ ...s, ok: false }));
      }
    };
    tick();
    const t = setInterval(tick, ms);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [url, ms, enabled]);
  return state;
}

const zoneFrameUrl = (z, nonce, opts) => {
  const grid = Boolean(opts?.grid);
  const w = opts?.w ?? GRID_W;
  return `${API}/api/zones/${z}/frame.jpg?t=${encodeURIComponent(String(nonce))}${
    grid ? `&w=${encodeURIComponent(String(w))}` : ""
  }`;
};

function ZoneCardPlayback({ zid, paused }) {
  // Playback priority: HLS (hardware decoded) > MJPEG > blob polling
  // Blob polling always runs in background as safety net
  const [mode, setMode] = useState("hls"); // "hls" | "mjpeg" | "poll"
  const [hlsPlaying, setHlsPlaying] = useState(false);
  const [mjpegOk, setMjpegOk] = useState(false);
  const [blobUrl, setBlobUrl] = useState("");
  const blobUrlRef = useRef("");
  const timerRef = useRef(null);
  const videoRef = useRef(null);
  const hlsRef = useRef(null);

  const det = usePollJson(`${API}/api/zones/${zid}/detections`, 700, true, { count: 0, items: [] });
  const emergency = useMemo(() => {
    return (det.items || []).some((d) => {
      const s = String(d.label || "").toLowerCase();
      return s.includes("drown") || s.includes("emerg");
    });
  }, [det]);
  const personCount = useMemo(() => {
    return (det.items || []).filter((d) => String(d.label || "").toLowerCase() === "person").length;
  }, [det]);

  // HLS playback via hls.js
  useEffect(() => {
    if (mode !== "hls" || paused) {
      setHlsPlaying(false);
      return;
    }
    const video = videoRef.current;
    if (!video) { setMode("mjpeg"); return; }

    setHlsPlaying(false);
    const hlsUrl = `${API}/api/zones/${zid}/hls/stream.m3u8`;

    // Watchdog: if video hasn't started playing within 5s → MJPEG
    let playing = false;
    const onTimeUpdate = () => {
      if (!playing) { playing = true; setHlsPlaying(true); }
    };
    video.addEventListener("timeupdate", onTimeUpdate);
    const watchdog = setTimeout(() => {
      if (!playing) {
        console.warn(`[HLS] Zone ${zid}: no playback after 5s, falling back to MJPEG`);
        setMode("mjpeg");
      }
    }, 5000);

    // Safari native HLS
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = hlsUrl;
      video.play().catch(() => {});
      const onError = () => setMode("mjpeg");
      video.addEventListener("error", onError);
      return () => {
        video.removeEventListener("error", onError);
        video.removeEventListener("timeupdate", onTimeUpdate);
        clearTimeout(watchdog);
        video.src = "";
      };
    }

    // hls.js
    if (!Hls.isSupported()) { setMode("mjpeg"); clearTimeout(watchdog); return; }

    const hls = new Hls({
      liveSyncDurationCount: 2,
      liveMaxLatencyDurationCount: 4,
      liveDurationInfinity: true,
      enableWorker: true,
      lowLatencyMode: true,
      maxBufferLength: 4,
      maxMaxBufferLength: 8,
      maxBufferSize: 2 * 1024 * 1024,
      manifestLoadingTimeOut: 6000,
      manifestLoadingMaxRetry: 2,
      manifestLoadingRetryDelay: 1000,
      levelLoadingTimeOut: 6000,
      fragLoadingTimeOut: 6000,
    });
    hlsRef.current = hls;

    let retryCount = 0;
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      video.play().catch(() => {});
    });

    hls.on(Hls.Events.ERROR, (_event, data) => {
      if (data.fatal) {
        if (data.type === Hls.ErrorTypes.NETWORK_ERROR && retryCount < 2) {
          retryCount++;
          setTimeout(() => hls.loadSource(hlsUrl), 1500);
        } else {
          hls.destroy();
          hlsRef.current = null;
          setMode("mjpeg");
        }
      }
    });

    hls.loadSource(hlsUrl);
    hls.attachMedia(video);

    return () => {
      clearTimeout(watchdog);
      video.removeEventListener("timeupdate", onTimeUpdate);
      hls.destroy();
      hlsRef.current = null;
    };
  }, [zid, paused, mode]);

  // MJPEG watchdog: if it doesn't load in 3s, fall to poll
  useEffect(() => {
    if (mode !== "mjpeg" || paused) return;
    setMjpegOk(false);
    const t = setTimeout(() => {
      if (!mjpegOk) setMode("poll");
    }, 3000);
    return () => clearTimeout(t);
  }, [zid, paused, mode, mjpegOk]);

  // Blob polling — ALWAYS runs as background safety net
  useEffect(() => {
    if (paused) return; // only stop when paused

    let alive = true;
    let inFlight = false;
    const ctrl = new AbortController();

    const fetchFrame = async () => {
      if (!alive || inFlight) return;
      inFlight = true;
      try {
        const r = await fetch(zoneFrameUrl(zid, Date.now(), { grid: true, w: GRID_W }), { cache: "no-store", signal: ctrl.signal });
        if (!r.ok) return;
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        if (!alive) {
          URL.revokeObjectURL(url);
          return;
        }
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = url;
        setBlobUrl(url);
      } catch {
      } finally {
        inFlight = false;
      }
    };

    fetchFrame();
    // Poll slower when HLS/MJPEG is active (just for safety), faster when it's the primary
    const interval = mode === "poll" ? FALLBACK_FRAME_MS : 1000;
    timerRef.current = setInterval(fetchFrame, interval);

    return () => {
      alive = false;
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      ctrl.abort();
    };
  }, [zid, paused, mode]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = "";
    };
  }, []);

  const height = "100%";

  return (
    <Box sx={{ position: "relative", width: "100%", height }}>
      {/* Blob polling background — always present as safety net */}
      <img
        src={blobUrl}
        alt={`Zone ${zid} frame`}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          background: "#0a2e38",
          userSelect: "none",
          pointerEvents: "none",
          display: blobUrl ? "block" : "none",
        }}
      />

      {/* HLS <video> - hardware decoded, overlays blob when playing */}
      <video
        ref={videoRef}
        muted
        autoPlay
        playsInline
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          background: "transparent",
          display: !paused && mode === "hls" && hlsPlaying ? "block" : "none",
        }}
      />

      {/* MJPEG fallback */}
      <img
        src={!paused && mode === "mjpeg" ? `${API}/api/zones/${zid}/stream.mjpg` : ""}
        alt={`Zone ${zid} stream`}
        onLoad={() => setMjpegOk(true)}
        onError={() => {
          setMode("poll");
          setMjpegOk(false);
        }}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          background: "transparent",
          userSelect: "none",
          pointerEvents: "none",
          display: !paused && mode === "mjpeg" ? "block" : "none",
        }}
      />

      {/* Live indicator */}
      <Box sx={{ position: "absolute", top: 10, left: 10, zIndex: 3, display: "flex", gap: 1 }}>
        <Chip
          icon={<FiberManualRecordIcon sx={{ fontSize: 10, color: paused ? "#ffa726" : "#4caf50", animation: paused ? "none" : "pulse 1.5s infinite", "@keyframes pulse": { "0%, 100%": { opacity: 1 }, "50%": { opacity: 0.4 } } }} />}
          label={paused ? "PAUSED" : "LIVE"}
          size="small"
          sx={{ bgcolor: "rgba(0,0,0,.75)", color: "#fff", fontWeight: 800, fontSize: 11, height: 24, "& .MuiChip-icon": { ml: 0.5 } }}
        />
      </Box>

      {/* Emergency alert */}
      {emergency && (
        <Box sx={{ position: "absolute", top: 10, right: 10, zIndex: 3, animation: "blink 0.8s infinite", "@keyframes blink": { "0%, 100%": { opacity: 1 }, "50%": { opacity: 0.5 } } }}>
          <Chip
            icon={<WarningAmberIcon sx={{ fontSize: 16 }} />}
            label="EMERGENCY"
            size="small"
            sx={{ bgcolor: "#d32f2f", color: "#fff", fontWeight: 900, "& .MuiChip-icon": { color: "#fff" } }}
          />
        </Box>
      )}

      {/* Detection stats overlay */}
      <Box sx={{ position: "absolute", left: 10, bottom: 10, zIndex: 3, display: "flex", gap: 1 }}>
        <Chip
          icon={<PersonIcon sx={{ fontSize: 14 }} />}
          label={personCount}
          size="small"
          sx={{ bgcolor: "rgba(0,0,0,.75)", color: "#4fc3f7", fontWeight: 800, fontSize: 12, height: 24, "& .MuiChip-icon": { color: "#4fc3f7" } }}
        />
        <Chip
          label={`${det.count ?? 0} detections`}
          size="small"
          sx={{ bgcolor: emergency ? "rgba(211,47,47,.9)" : "rgba(0,0,0,.75)", color: "#fff", fontWeight: 800, fontSize: 12, height: 24 }}
        />
      </Box>
    </Box>
  );
}


// Voice alert for emergencies with speech synthesis - with smart rate limiting

// --- Preload voices (Chrome loads them asynchronously) ---
let _cachedVoices = [];
const _loadVoices = () => {
  if ('speechSynthesis' in window) {
    _cachedVoices = window.speechSynthesis.getVoices();
  }
};
_loadVoices();
if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
  window.speechSynthesis.onvoiceschanged = _loadVoices;
}

// Pick best deep male voice available
const getMaleVoice = () => {
  const voices = _cachedVoices.length > 0 ? _cachedVoices : (window.speechSynthesis?.getVoices() || []);
  // Priority order: deep male voices
  const maleKeywords = ['David', 'Mark', 'James', 'Daniel', 'Male', 'Guy', 'Richard', 'George'];
  for (const keyword of maleKeywords) {
    const found = voices.find(v => v.name.includes(keyword) && v.lang.startsWith('en'));
    if (found) return found;
  }
  // Fallback: any English Google voice (tends to sound professional)
  const googleEn = voices.find(v => v.name.includes('Google') && v.lang.startsWith('en'));
  if (googleEn) return googleEn;
  // Fallback: any English voice
  const anyEn = voices.find(v => v.lang.startsWith('en'));
  if (anyEn) return anyEn;
  return null;
};

const useEmergencyVoiceAlert = (alerts, soundEnabled, openZone, paused) => {
  const seenAlertsRef = useRef(new Set()); // Track alerts we've already announced
  const lastPlayRef = useRef(0);
  const speakingRef = useRef(false);
  const alertCountRef = useRef(0); // Track alerts in current window
  const windowStartRef = useRef(Date.now());
  
  useEffect(() => {
    if (!soundEnabled || paused) return;
    
    // Find emergency alerts
    const emergencyAlerts = (alerts || []).filter((a) => {
      const l = String(a.label || "").toLowerCase();
      return l.includes("drown") || l.includes("emerg");
    });
    
    if (emergencyAlerts.length === 0) return;
    
    const now = Date.now();
    
    // Reset counter every 60 seconds
    if (now - windowStartRef.current > 60000) {
      alertCountRef.current = 0;
      windowStartRef.current = now;
    }
    
    // Maximum 2 voice alerts per minute to prevent spam
    if (alertCountRef.current >= 2) return;
    
    // Minimum 30 seconds between any voice alerts
    if (now - lastPlayRef.current < 30000) return;
    
    // Check if this is a truly NEW alert we haven't announced
    const latestAlert = emergencyAlerts[0];
    // Create a unique ID from timestamp + zone + label
    const alertId = `${latestAlert.timestamp || latestAlert.ts}_${latestAlert.zone}_${latestAlert.label}`;
    
    // If we've already seen this exact alert, skip
    if (seenAlertsRef.current.has(alertId)) return;
    
    // Mark as seen and update counters
    seenAlertsRef.current.add(alertId);
    lastPlayRef.current = now;
    alertCountRef.current++;
    
    // Keep seen alerts set from growing too large (only last 100)
    if (seenAlertsRef.current.size > 100) {
      const arr = Array.from(seenAlertsRef.current);
      seenAlertsRef.current = new Set(arr.slice(-50));
    }
    
    // Play alarm sound (shorter, less intrusive)
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      for (let i = 0; i < 2; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.setValueAtTime(800, ctx.currentTime + i * 0.25);
        osc.frequency.setValueAtTime(600, ctx.currentTime + i * 0.25 + 0.1);
        gain.gain.setValueAtTime(0.3, ctx.currentTime + i * 0.25);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + i * 0.25 + 0.2);
        osc.start(ctx.currentTime + i * 0.25);
        osc.stop(ctx.currentTime + i * 0.25 + 0.2);
      }
    } catch (e) {}
    
    // Speak voice announcement
    if ('speechSynthesis' in window && !speakingRef.current) {
      speakingRef.current = true;
      window.speechSynthesis.cancel();
      
      const zone = latestAlert.zone || openZone;
      const label = String(latestAlert.label || "drowning").toLowerCase();
      let message = label.includes("drown") 
        ? (zone ? `Alert! Drowning detected in Zone ${zone}. Please check immediately.` : "Alert! Drowning detected. Please check immediately.")
        : (zone ? `Emergency alert in Zone ${zone}. Please respond.` : "Emergency alert detected. Please respond.");
      
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 1.0;
      utterance.volume = 1.0;
      utterance.pitch = 0.9;
      utterance.lang = 'en-US';
      
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = getMaleVoice();
      if (preferredVoice) utterance.voice = preferredVoice;
      
      utterance.onend = () => { speakingRef.current = false; };
      utterance.onerror = () => { speakingRef.current = false; };
      
      setTimeout(() => window.speechSynthesis.speak(utterance), 500);
    }
  }, [alerts, soundEnabled, openZone, paused]);
};

// Speak function for announcements
const speakAnnouncement = (message, rate = 0.9) => {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = rate;
    utterance.volume = 1.0;
    utterance.lang = 'en-US';
    utterance.pitch = 0.9;
    const preferredVoice = getMaleVoice();
    if (preferredVoice) utterance.voice = preferredVoice;
    window.speechSynthesis.speak(utterance);
  }
};

// ===================== PERSON COUNT TIMELINE COMPONENT =====================
function PersonCountTimeline({ api, zoneId, zoneName }) {
  const timelineData = usePollJson(`${api}/api/zones/${zoneId}/timeline`, 5000, true, { timeline: [] });
  const colors = ["#2dd4bf","#34d399","#f59e0b","#a78bfa","#f472b6","#22d3ee"];
  const accentColor = colors[((zoneId || 1) - 1) % colors.length];

  const { chartData, currentCount, peakCount, avgCount, dataPoints, trendDirection } = useMemo(() => {
    const empty = {
      chartData: { labels: [], datasets: [{ label: "People", data: [], borderColor: accentColor, backgroundColor: "transparent", tension: 0.35 }] },
      currentCount: 0, peakCount: 0, avgCount: 0, dataPoints: 0, trendDirection: "stable",
    };
    if (!timelineData?.timeline?.length) return empty;

    const now = Date.now();
    const last24h = timelineData.timeline.filter(p => (now - p.timestamp * 1000) < 24 * 60 * 60 * 1000);
    if (!last24h.length) return empty;

    // Keep every 2nd-3rd point to reduce clutter
    const stride = Math.max(1, Math.floor(last24h.length / 20));
    const filteredData = last24h.filter((_, idx) => idx % stride === 0 || idx === last24h.length - 1);
    
    const counts = filteredData.map(p => p.count);
    const peak = Math.max(...last24h.map(p => p.count));
    const avg = last24h.reduce((a, b) => a + b.count, 0) / last24h.length;
    const current = last24h[last24h.length - 1].count;
    
    // Calculate trend
    let trend = "stable";
    if (last24h.length >= 2) {
      const recent = last24h.slice(-5).map(p => p.count);
      const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
      const older = last24h.slice(-10, -5).map(p => p.count);
      const olderAvg = older.length ? older.reduce((a, b) => a + b, 0) / older.length : recentAvg;
      if (recentAvg > olderAvg * 1.1) trend = "up";
      else if (recentAvg < olderAvg * 0.9) trend = "down";
    }

    return {
      chartData: {
        labels: filteredData.map(p => new Date(p.timestamp * 1000)),
        datasets: [{
          label: "People Detected",
          data: counts,
          borderColor: accentColor,
          backgroundColor: (ctx) => {
            if (!ctx.chart.chartArea) return "transparent";
            const { top, bottom } = ctx.chart.chartArea;
            const grad = ctx.chart.ctx.createLinearGradient(0, top, 0, bottom);
            grad.addColorStop(0, accentColor + "70");
            grad.addColorStop(0.5, accentColor + "40");
            grad.addColorStop(1, "transparent");
            return grad;
          },
          borderWidth: 3.5,
          tension: 0.45,
          fill: true,
          pointRadius: 4,
          pointHoverRadius: 8,
          pointBackgroundColor: accentColor,
          pointBorderColor: "#fff",
          pointBorderWidth: 2.5,
          pointHoverBackgroundColor: "#fff",
          pointHoverBorderColor: accentColor,
          pointHoverBorderWidth: 3,
          segment: {
            borderColor: (ctx) => ctx.p0DataIndex === undefined ? accentColor : accentColor + "dd",
          },
        }],
      },
      currentCount: current,
      peakCount: peak,
      avgCount: avg,
      dataPoints: last24h.length,
      trendDirection: trend,
    };
  }, [timelineData, zoneName, accentColor]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 1000,
      easing: "easeInOutQuart",
      delay: (ctx) => {
        let delay = 0;
        if (ctx.type === "data") {
          delay = ctx.dataIndex * 20 + ctx.datasetIndex * 100;
        }
        return delay;
      },
    },
    animations: {
      tension: { duration: 1000, from: 0.1, to: 0.3, loop: false },
      fill: { duration: 1000 },
      radius: { duration: 1000 },
    },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      title: { display: false },
      tooltip: {
        backgroundColor: "rgba(10,14,19,0.95)",
        titleColor: "#fff",
        bodyColor: "rgba(255,255,255,0.85)",
        borderColor: "rgba(255,255,255,0.1)",
        borderWidth: 1,
        padding: 12,
        cornerRadius: 8,
        titleFont: { size: 12, weight: "600" },
        bodyFont: { size: 11, weight: "500" },
        displayColors: false,
        boxPadding: 6,
        callbacks: {
          title: (items) => {
            if (!items.length) return "";
            const d = new Date(items[0].parsed.x);
            return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
          },
          label: (item) => `People: ${item.parsed.y}`,
        },
      },
    },
    scales: {
      x: {
        type: "time",
        time: { unit: "minute", stepSize: 15, displayFormats: { minute: "HH:mm", hour: "HH:mm" } },
        title: { display: true, text: "Time", color: "rgba(255,255,255,0.5)", font: { size: 12, weight: "600" }, padding: { top: 12 } },
        ticks: { color: "rgba(255,255,255,0.5)", font: { size: 11, weight: "500" }, maxRotation: 0, autoSkip: true, maxTicksLimit: 5 },
        grid: { color: "rgba(255,255,255,0.08)", drawBorder: false, lineWidth: 1, drawTicks: false },
        border: { display: false },
      },
      y: {
        beginAtZero: true,
        title: { display: true, text: "People Count", color: "rgba(255,255,255,0.5)", font: { size: 12, weight: "600" }, padding: { bottom: 12 } },
        ticks: {
          color: "rgba(255,255,255,0.5)",
          font: { size: 11, weight: "500" },
          precision: 0,
          padding: 10,
          callback: (v) => Number.isInteger(v) ? v : "",
        },
        grid: { color: "rgba(255,255,255,0.08)", drawBorder: false, lineWidth: 1, drawTicks: false },
        border: { display: false },
      },
    },
  }), [accentColor]);

  return (
    <Box sx={{
      borderRadius: 3,
      background: "#0f1419",
      border: `1px solid rgba(255,255,255,0.08)`,
      overflow: "hidden",
      boxShadow: "0 4px 16px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)",
      transition: "all 0.3s ease",
      position: "relative",
      "&:before": {
        content: '""',
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        height: "1px",
        background: `linear-gradient(90deg, transparent, ${accentColor}80, transparent)`,
        opacity: 0.6,
      },
      "&:hover": {
        boxShadow: "0 8px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08)",
        border: `1px solid rgba(255,255,255,0.12)`,
      },
    }}>
      {/* Card Header */}
      <Box sx={{ px: 4, pt: 3, pb: 2, display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(255,255,255,0.01)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2.5 }}>
          <Box sx={{
            width: 12,
            height: 12,
            borderRadius: "50%",
            bgcolor: accentColor,
            boxShadow: `0 0 12px ${accentColor}80`,
            animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
            "@keyframes pulse": {
              "0%, 100%": { opacity: 1, boxShadow: `0 0 12px ${accentColor}80` },
              "50%": { opacity: 0.6, boxShadow: `0 0 8px ${accentColor}40` },
            },
          }} />
          <Box>
            <Typography sx={{ fontWeight: 700, fontSize: 15, color: "#fff", letterSpacing: "-0.4px", lineHeight: 1 }}>{zoneName}</Typography>
            <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.45)", fontWeight: 500, mt: 0.2 }}>Person Count Timeline</Typography>
          </Box>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.8 }}>
            <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "#22c55e", animation: "pulse 1.2s cubic-bezier(0.4, 0, 0.6, 1) infinite", boxShadow: "0 0 6px #22c55e60" }} />
            <Typography sx={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.3px" }}>LIVE</Typography>
          </Box>
          <Chip
            label={`${dataPoints} points`}
            size="small"
            sx={{
              bgcolor: "rgba(255,255,255,0.05)",
              color: "rgba(255,255,255,0.7)",
              fontWeight: 600,
              fontSize: 11,
              height: 24,
              border: "1px solid rgba(255,255,255,0.1)",
              transition: "all 0.3s",
            }}
          />
        </Box>
      </Box>

      {/* Stats Row - Professional */}
      <Box sx={{ px: 4, py: 2.5, display: "flex", gap: 2, background: "rgba(255,255,255,0.01)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        {[
          { label: "Current", value: currentCount, icon: "👤", color: accentColor, trend: trendDirection },
          { label: "Peak", value: peakCount, icon: "📈", color: "#3b82f6" },
          { label: "Avg", value: avgCount.toFixed(1), icon: "📊", color: "#8b5cf6" },
        ].map((s, idx) => (
          <Box key={s.label} sx={{
            flex: 1,
            py: 2,
            px: 2.5,
            borderRadius: 2,
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            textAlign: "center",
            transition: "all 0.25s ease",
            cursor: "pointer",
            position: "relative",
            "&:before": {
              content: '""',
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              height: "1px",
              background: `linear-gradient(90deg, transparent, ${s.color}60, transparent)`,
              opacity: 0.4,
            },
            "&:hover": {
              background: "rgba(255,255,255,0.05)",
              border: `1px solid ${s.color}40`,
              boxShadow: `0 4px 12px ${s.color}15`,
            },
          }}>
            <Typography sx={{ fontSize: 18, mb: 0.5, display: "block" }}>{s.icon}</Typography>
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.4, mb: 0.2 }}>
              <Typography sx={{ 
                fontSize: 24, 
                fontWeight: 700, 
                color: s.color, 
                lineHeight: 1,
                animation: "numPulse 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
                "@keyframes numPulse": {
                  "0%": { transform: "scale(1.1)", opacity: 0.8 },
                  "100%": { transform: "scale(1)", opacity: 1 },
                },
              }}>
                {s.value}
              </Typography>
              {s.trend === "up" && <Typography sx={{ fontSize: 16, color: "#f59e0b", animation: "bounce 1s infinite" }}>📈</Typography>}
              {s.trend === "down" && <Typography sx={{ fontSize: 16, color: "#ff5252" }}>📉</Typography>}
            </Box>
            <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.3px" }}>
              {s.label}
            </Typography>
          </Box>
        ))}
      </Box>

      {/* Chart Container - Professional Grid Design */}
      <Box sx={{ 
        px: 4, 
        py: 4, 
        minHeight: 320, 
        position: "relative", 
        background: "#0a0e13",
        backgroundImage: `
          linear-gradient(0deg, rgba(74,222,128,0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(74,222,128,0.03) 1px, transparent 1px)
        `,
        backgroundSize: '40px 40px',
        borderTop: "1px solid rgba(255,255,255,0.06)",
      }}>
        {dataPoints > 0 ? (
          <Box sx={{ 
            height: 300, 
            position: "relative",
            animation: "chartSlideIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
            "@keyframes chartSlideIn": {
              "0%": { opacity: 0.7, transform: "scale(0.98)" },
              "100%": { opacity: 1, transform: "scale(1)" },
            },
          }}>
            <Box sx={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: "transparent",
              pointerEvents: "none",
              zIndex: 1,
            }} />
            <Line data={chartData} options={chartOptions} key={`${chartData.labels.length}`} />
          </Box>
        ) : (
          <Box sx={{ height: 300, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 2 }}>
            <Box sx={{
              width: 60,
              height: 60,
              borderRadius: "50%",
              background: "rgba(74,222,128,0.1)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              animation: "float 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
              boxShadow: "0 0 16px rgba(74,222,128,0.2)",
              "@keyframes float": {
                "0%, 100%": { transform: "translateY(0px)" },
                "50%": { transform: "translateY(-6px)" },
              },
            }}>
              <TrendingUpIcon sx={{ fontSize: 32, color: "#22c55e" }} />
            </Box>
            <Typography sx={{ color: "rgba(255,255,255,0.8)", fontSize: 16, fontWeight: 600 }}>Collecting Live Data</Typography>
            <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 12, fontWeight: 500, textAlign: "center", maxWidth: "85%" }}>Person counts will appear as detections happen in real-time</Typography>
          </Box>
        )
      }
      </Box>

      {/* Footer Insights */}
      {dataPoints > 0 && (
        <Box sx={{ px: 4, py: 2.5, background: "rgba(255,255,255,0.01)", borderTop: "1px solid rgba(255,255,255,0.06)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Box sx={{ height: 2, width: 20, borderRadius: 1, background: "rgba(74,222,128,0.4)", boxShadow: "0 0 8px rgba(74,222,128,0.2)" }} />
            <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontWeight: 500 }}>Timeline Chart</Typography>
          </Box>
          <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>
            <Chip
              label={trendDirection === "up" ? "Trending Up" : trendDirection === "down" ? "Trending Down" : "Stable"}
              size="small"
              sx={{
                bgcolor: trendDirection === "up" ? "rgba(59,130,246,0.1)" : trendDirection === "down" ? "rgba(255,82,82,0.1)" : "rgba(34,197,94,0.1)",
                color: trendDirection === "up" ? "#3b82f6" : trendDirection === "down" ? "#ff5252" : "#22c55e",
                fontWeight: 600,
                fontSize: 10,
                height: 24,
                border: `1px solid ${trendDirection === "up" ? "rgba(59,130,246,0.3)" : trendDirection === "down" ? "rgba(255,82,82,0.3)" : "rgba(34,197,94,0.3)"}`,
              }}
            />
          </Box>
        </Box>
      )}
    </Box>
  );
}

// ===================== ZONE NAME EDITOR COMPONENT =====================
function ZoneNameEditor({ zoneId, currentName, api, onNameChanged }) {
  const [open, setOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch(`${api}/api/zones/${zoneId}/name`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName.trim() })
      });
      
      if (response.ok) {
        const data = await response.json();
        if (onNameChanged) onNameChanged(data.name);
        setOpen(false);
      }
    } catch (error) {
      console.error('Failed to save zone name:', error);
    }
    setSaving(false);
  };

  const handleOpen = () => {
    setNewName(currentName.startsWith('Zone ') ? '' : currentName);
    setOpen(true);
  };

  return (
    <>
      <Chip
        icon={<EditIcon sx={{ fontSize: 14 }} />}
        label={currentName}
        onClick={handleOpen}
        sx={{
          bgcolor: "rgba(45,212,191,0.15)",
          color: "#2dd4bf",
          fontWeight: 700,
          cursor: "pointer",
          "&:hover": { bgcolor: "rgba(45,212,191,0.25)" }
        }}
      />
      
      <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ bgcolor: "#0c1621", color: "#fff" }}>
          Rename Zone {zoneId}
        </DialogTitle>
        <DialogContent sx={{ bgcolor: "#0c1621", pt: 3 }}>
          <TextField
            autoFocus
            fullWidth
            label="Zone Name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="e.g., North Beach, Swimming Pool"
            sx={{
              "& .MuiOutlinedInput-root": {
                color: "#fff",
                "& fieldset": { borderColor: "rgba(255,255,255,0.3)" },
                "&:hover fieldset": { borderColor: "#2dd4bf" },
                "&.Mui-focused fieldset": { borderColor: "#2dd4bf" }
              },
              "& .MuiInputLabel-root": { color: "rgba(255,255,255,0.7)" }
            }}
          />
        </DialogContent>
        <DialogActions sx={{ bgcolor: "#0c1621", p: 3 }}>
          <Button onClick={() => setOpen(false)} sx={{ color: "rgba(255,255,255,0.7)" }}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={saving || !newName.trim()}
            variant="contained"
            sx={{
              bgcolor: "#2dd4bf",
              color: "#000",
              fontWeight: 700,
              "&:hover": { bgcolor: "#14b8a6" }
            }}
          >
            {saving ? "Saving..." : "Save"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

// ===================== VIDEO MANAGER COMPONENT =====================
function VideoManager({ api, onReload }) {
  const [videos, setVideos] = useState([]);
  const [videoDir, setVideoDir] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [renaming, setRenaming] = useState(null); // filename being renamed
  const [newName, setNewName] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const fetchVideos = useCallback(async () => {
    try {
      const r = await fetch(`${api}/api/videos`);
      if (r.ok) {
        const data = await r.json();
        setVideos(data.items || []);
        setVideoDir(data.video_dir || "");
      }
    } catch {}
  }, [api]);

  useEffect(() => { fetchVideos(); const t = setInterval(fetchVideos, 3000); return () => clearInterval(t); }, [fetchVideos]);

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadProgress(`Uploading ${files.length} file(s)...`);
    try {
      const fd = new FormData();
      for (const f of files) fd.append("file", f);
      const r = await fetch(`${api}/api/videos/upload`, { method: "POST", body: fd });
      const data = await r.json();
      if (r.ok) {
        const ok = (data.uploaded || []).filter((u) => u.ok).length;
        const fail = (data.uploaded || []).filter((u) => !u.ok).length;
        setUploadProgress(`Uploaded ${ok} file(s)${fail > 0 ? `, ${fail} failed` : ""}. Zones reloaded.`);
      } else {
        setUploadProgress(data.error || "Upload failed");
      }
    } catch (e) {
      setUploadProgress(`Error: ${e.message}`);
    }
    setUploading(false);
    fetchVideos();
    if (onReload) onReload();
    setTimeout(() => setUploadProgress(""), 4000);
  };

  const handleDelete = async (filename) => {
    if (!window.confirm(`Delete "${filename}"? This will stop its zone and remove the file.`)) return;
    try {
      const r = await fetch(`${api}/api/videos/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (r.ok) {
        fetchVideos();
        if (onReload) onReload();
      }
    } catch {}
  };

  const handleRename = async (oldName) => {
    if (!newName.trim() || newName.trim() === oldName) { setRenaming(null); return; }
    try {
      const r = await fetch(`${api}/api/videos/${encodeURIComponent(oldName)}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: newName.trim() }),
      });
      if (r.ok) {
        fetchVideos();
        if (onReload) onReload();
      } else {
        const data = await r.json();
        alert(data.error || "Rename failed");
      }
    } catch {}
    setRenaming(null);
    setNewName("");
  };

  const onDrop = (e) => { e.preventDefault(); setDragOver(false); handleUpload(e.dataTransfer.files); };
  const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);

  return (
    <Box>
      <Typography sx={{ fontWeight: 900, fontSize: 26, color: "#fff", display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
        <VideoLibraryIcon sx={{ color: "#2dd4bf", fontSize: 32 }} />
        Video Manager
      </Typography>
      <Typography sx={{ fontSize: 15, color: "rgba(255,255,255,0.5)", mb: 4 }}>
        Upload, manage and organize surveillance video clips. Any video file added here automatically becomes a zone.
      </Typography>

      {/* Upload Area */}
      <Box
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => fileInputRef.current?.click()}
        sx={{
          p: 5,
          mb: 4,
          borderRadius: 4,
          border: dragOver ? "3px dashed #2dd4bf" : "3px dashed rgba(255,255,255,0.12)",
          bgcolor: dragOver ? "rgba(45,212,191,0.08)" : "#0f1923",
          cursor: "pointer",
          textAlign: "center",
          transition: "all 0.3s ease",
          "&:hover": { borderColor: "rgba(45,212,191,0.5)", bgcolor: "rgba(45,212,191,0.05)" },
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          multiple
          hidden
          onChange={(e) => handleUpload(e.target.files)}
        />
        <CloudUploadIcon sx={{ fontSize: 56, color: dragOver ? "#2dd4bf" : "rgba(255,255,255,0.3)", mb: 2 }} />
        <Typography sx={{ fontWeight: 800, fontSize: 20, color: dragOver ? "#2dd4bf" : "#fff", mb: 1 }}>
          {uploading ? "Uploading..." : "Drop video files here or click to browse"}
        </Typography>
        <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 14 }}>
          Supports MP4, AVI, MKV, MOV, WebM, FLV, WMV, M4V, TS
        </Typography>
        {uploadProgress && (
          <Chip
            icon={<CheckCircleIcon sx={{ fontSize: 16 }} />}
            label={uploadProgress}
            sx={{ mt: 2, bgcolor: "rgba(45,212,191,0.15)", color: "#2dd4bf", fontWeight: 700 }}
          />
        )}
        {uploading && <LinearProgress sx={{ mt: 2, borderRadius: 2, bgcolor: "rgba(45,212,191,0.15)", "& .MuiLinearProgress-bar": { bgcolor: "#2dd4bf" } }} />}
      </Box>

      {/* Video directory info */}
      {videoDir && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 3, px: 1 }}>
          <FolderOpenIcon sx={{ color: "rgba(255,255,255,0.3)", fontSize: 18 }} />
          <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.4)", fontFamily: "monospace" }}>{videoDir}</Typography>
        </Box>
      )}

      {/* Video Files List */}
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr", lg: "1fr 1fr 1fr" }, gap: 3 }}>
        {videos.length === 0 ? (
          <Box sx={{ gridColumn: "1/-1", py: 10, textAlign: "center", borderRadius: 4, bgcolor: "#0f1923", border: "2px dashed rgba(255,255,255,0.08)" }}>
            <VideoLibraryIcon sx={{ fontSize: 50, color: "rgba(255,255,255,0.15)", mb: 2 }} />
            <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 16 }}>No video files found</Typography>
            <Typography sx={{ color: "rgba(255,255,255,0.25)", fontSize: 13, mt: 1 }}>Upload videos above to create surveillance zones</Typography>
          </Box>
        ) : (
          videos.map((v) => (
            <Box
              key={v.filename}
              sx={{
                p: 3,
                borderRadius: 3,
                bgcolor: "#0f1923",
                border: v.active ? "1px solid rgba(45,212,191,0.25)" : "1px solid rgba(255,255,255,0.06)",
                transition: "all 0.3s ease",
                "&:hover": { borderColor: "rgba(255,255,255,0.15)", boxShadow: "0 8px 30px rgba(0,0,0,0.3)" },
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}>
                <Avatar sx={{
                  background: v.active ? "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)" : "rgba(255,255,255,0.08)",
                  width: 44, height: 44, fontSize: 16, fontWeight: 900,
                  color: v.active ? "#071520" : "rgba(255,255,255,0.4)",
                }}>
                  {v.zone_id || "?"}
                </Avatar>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  {renaming === v.filename ? (
                    <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                      <input
                        autoFocus
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") handleRename(v.filename); if (e.key === "Escape") setRenaming(null); }}
                        style={{
                          background: "rgba(255,255,255,0.08)", border: "1px solid rgba(45,212,191,0.4)", borderRadius: 6,
                          color: "#fff", padding: "6px 10px", fontSize: 14, fontWeight: 600, width: "100%", outline: "none",
                        }}
                      />
                      <IconButton size="small" onClick={() => handleRename(v.filename)} sx={{ color: "#2dd4bf" }}>
                        <CheckCircleIcon sx={{ fontSize: 20 }} />
                      </IconButton>
                    </Box>
                  ) : (
                    <Typography sx={{ fontWeight: 700, fontSize: 15, color: "#fff", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={v.filename}>
                      {v.filename}
                    </Typography>
                  )}
                  <Box sx={{ display: "flex", gap: 1.5, mt: 0.5, alignItems: "center" }}>
                    <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.4)" }}>{v.size_mb} MB</Typography>
                    {v.zone_id && <Chip label={`Zone ${v.zone_id}`} size="small" sx={{ height: 20, fontSize: 10, fontWeight: 800, bgcolor: "rgba(45,212,191,0.12)", color: "#2dd4bf" }} />}
                    <Chip
                      label={v.active ? "ACTIVE" : "IDLE"}
                      size="small"
                      sx={{
                        height: 20, fontSize: 10, fontWeight: 800,
                        bgcolor: v.active ? "rgba(105,240,174,0.15)" : "rgba(255,255,255,0.05)",
                        color: v.active ? "#69f0ae" : "rgba(255,255,255,0.35)",
                      }}
                    />
                  </Box>
                </Box>
              </Box>

              {/* Actions */}
              <Box sx={{ display: "flex", gap: 1, justifyContent: "flex-end" }}>
                <Tooltip title="Rename">
                  <IconButton
                    size="small"
                    onClick={() => { setRenaming(v.filename); setNewName(v.filename); }}
                    sx={{ color: "rgba(255,255,255,0.4)", "&:hover": { color: "#2dd4bf", bgcolor: "rgba(45,212,191,0.1)" } }}
                  >
                    <EditIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Delete">
                  <IconButton
                    size="small"
                    onClick={() => handleDelete(v.filename)}
                    sx={{ color: "rgba(255,255,255,0.4)", "&:hover": { color: "#ff5252", bgcolor: "rgba(255,82,82,0.1)" } }}
                  >
                    <DeleteIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          ))
        )}
      </Box>

      {/* Reload button */}
      <Box sx={{ mt: 4, display: "flex", gap: 2 }}>
        <Button
          startIcon={<RefreshIcon />}
          variant="contained"
          onClick={() => { onReload?.(); fetchVideos(); }}
          sx={{ background: "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)", color: "#071520", fontWeight: 800, px: 4, py: 1.2, borderRadius: "12px", textTransform: "none", boxShadow: "0 4px 24px rgba(45,212,191,0.3)", "&:hover": { background: "linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)" } }}
        >
          Reload All Zones
        </Button>
      </Box>
    </Box>
  );
}

export default function App() {
  const [tab, setTab] = useState(0);
  const [paused, setPaused] = useState(false);
  const [openZone, setOpenZone] = useState(null);
  const [analyticsSection, setAnalyticsSection] = useState("overview");
  
  // Quick win states
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoAnnounce, setAutoAnnounce] = useState(true); // Auto announce toggle - ON by default for voice alerts

  const [modalPaused, setModalPaused] = useState(false);
  const [modalMode, setModalMode] = useState("hls"); // "hls" | "mjpeg" | "poll"
  const [modalHlsPlaying, setModalHlsPlaying] = useState(false);
  const [modalMjpegOk, setModalMjpegOk] = useState(false);

  const [modalBlobUrl, setModalBlobUrl] = useState("");
  const [zoneNames, setZoneNames] = useState(new Map());
  const modalBlobUrlRef = useRef("");
  const modalVideoBoxRef = useRef(null);
  const modalVideoRef = useRef(null);
  const modalHlsRef = useRef(null);

  const { data: health, ok: backendOk } = usePollJsonWithOk(`${API}/api/health`, 2000, true, {
    status: "unknown",
    device: "?",
    model_names: [],
  });

  const zonesResp = usePollJson(`${API}/api/zones`, 1200, true, { items: [] });
  const zones = useMemo(() => (zonesResp.items || []).map((x) => x.id).filter((x) => Number.isFinite(x)), [zonesResp]);
  const zoneMeta = useMemo(() => new Map((zonesResp.items || []).map((x) => [x.id, x])), [zonesResp]);

  // Update zone names when zones data changes
  useEffect(() => {
    if (zonesResp.items) {
      const newNames = new Map();
      zonesResp.items.forEach(item => {
        if (item.name) {
          newNames.set(item.id, item.name);
        }
      });
      setZoneNames(newNames);
    }
  }, [zonesResp.items]);

  useEffect(() => {
    // FIX: don't auto-close the modal while zones are still loading (zones can be [] briefly)
    if (openZone == null) return;
    if (!Array.isArray(zones) || zones.length === 0) return;
    if (!zones.includes(openZone)) setOpenZone(null);
  }, [openZone, zones]);

  // Keep wheel/pinch inside modal box (don’t scroll page)
  useEffect(() => {
    const el = modalVideoBoxRef.current;
    if (!el) return;
    const onWheel = (e) => e.preventDefault();
    const onTouchMove = (e) => e.preventDefault();
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchmove", onTouchMove);
    };
  }, [openZone]);


  // Modal HLS playback with watchdog
  useEffect(() => {
    if (!openZone || modalMode !== "hls" || modalPaused) {
      setModalHlsPlaying(false);
      return;
    }
    const video = modalVideoRef.current;
    if (!video) { setModalMode("mjpeg"); return; }

    setModalHlsPlaying(false);
    const hlsUrl = `${API}/api/zones/${openZone}/hls/stream.m3u8`;

    // Watchdog: if no playback in 5s → MJPEG
    let playing = false;
    const onTimeUpdate = () => {
      if (!playing) { playing = true; setModalHlsPlaying(true); }
    };
    video.addEventListener("timeupdate", onTimeUpdate);
    const watchdog = setTimeout(() => {
      if (!playing) {
        console.warn(`[HLS] Modal zone ${openZone}: no playback, falling back`);
        setModalMode("mjpeg");
      }
    }, 5000);

    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = hlsUrl;
      video.play().catch(() => {});
      const onError = () => setModalMode("mjpeg");
      video.addEventListener("error", onError);
      return () => {
        video.removeEventListener("error", onError);
        video.removeEventListener("timeupdate", onTimeUpdate);
        clearTimeout(watchdog);
        video.src = "";
      };
    }

    if (!Hls.isSupported()) { setModalMode("mjpeg"); clearTimeout(watchdog); return; }

    const hls = new Hls({
      liveSyncDurationCount: 2, liveMaxLatencyDurationCount: 4,
      liveDurationInfinity: true, enableWorker: true, lowLatencyMode: true,
      maxBufferLength: 4, maxMaxBufferLength: 8, maxBufferSize: 2 * 1024 * 1024,
      manifestLoadingTimeOut: 6000, manifestLoadingMaxRetry: 2,
      manifestLoadingRetryDelay: 1000, levelLoadingTimeOut: 6000, fragLoadingTimeOut: 6000,
    });
    modalHlsRef.current = hls;
    let retryCount = 0;
    hls.on(Hls.Events.MANIFEST_PARSED, () => { video.play().catch(() => {}); });
    hls.on(Hls.Events.ERROR, (_e, data) => {
      if (data.fatal) {
        if (data.type === Hls.ErrorTypes.NETWORK_ERROR && retryCount < 2) {
          retryCount++;
          setTimeout(() => hls.loadSource(hlsUrl), 1500);
        } else {
          hls.destroy(); modalHlsRef.current = null; setModalMode("mjpeg");
        }
      }
    });
    hls.loadSource(hlsUrl);
    hls.attachMedia(video);
    return () => {
      clearTimeout(watchdog);
      video.removeEventListener("timeupdate", onTimeUpdate);
      hls.destroy();
      modalHlsRef.current = null;
    };
  }, [openZone, modalPaused, modalMode]);
  // Modal: poll /frame.jpg - ALWAYS runs as background safety net
  useEffect(() => {
    if (!openZone || modalPaused) return;

    let alive = true;
    let inFlight = false;
    const ctrl = new AbortController();

    const fetchFrame = async () => {
      if (!alive || inFlight) return;
      inFlight = true;
      try {
        const r = await fetch(`${API}/api/zones/${openZone}/frame.jpg?t=${Date.now()}`, {
          cache: "no-store",
          signal: ctrl.signal,
        });
        if (!r.ok) return;
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        if (!alive) {
          URL.revokeObjectURL(url);
          return;
        }
        if (modalBlobUrlRef.current) URL.revokeObjectURL(modalBlobUrlRef.current);
        modalBlobUrlRef.current = url;
        setModalBlobUrl(url);
      } catch {
      } finally {
        inFlight = false;
      }
    };

    fetchFrame();
    const interval = modalMode === "poll" ? MODAL_REFRESH_MS : 1000;
    const t = setInterval(fetchFrame, interval);

    return () => {
      alive = false;
      clearInterval(t);
      ctrl.abort();
    };
  }, [openZone, modalPaused, modalMode]);

  useEffect(() => {
    if (!openZone) {
      setModalPaused(false);
      setModalMode("hls");
      setModalHlsPlaying(false);
      setModalMjpegOk(false);
      if (modalBlobUrlRef.current) URL.revokeObjectURL(modalBlobUrlRef.current);
      modalBlobUrlRef.current = "";
      setModalBlobUrl("");
    }
  }, [openZone]);

  const alerts = usePollJson(`${API}/api/alerts?limit=120`, 1000, true, { items: [] });
  const analysis = usePollJson(`${API}/api/analysis`, 1500, true, { alerts_total: 0, alerts_by_zone: {}, alerts_by_label: {} });

  const modalAlerts = usePollJson(openZone ? `${API}/api/alerts?zone=${openZone}&limit=40` : `${API}/api/alerts?limit=1`, 900, !!openZone, { items: [] });
  const modalAnalysis = usePollJson(openZone ? `${API}/api/analysis?zone=${openZone}` : `${API}/api/analysis`, 1200, !!openZone, { alerts_total: 0, alerts_by_zone: {}, alerts_by_label: {} });
  const modalDetections = usePollJson(openZone ? `${API}/api/zones/${openZone}/detections` : `${API}/api/zones/1/detections`, 250, !!openZone, { zone: null, count: 0, age_s: null, items: [] });

  const emergencyCount = useMemo(() => {
    return (alerts.items || []).filter((a) => {
      const l = String(a.label || "").toLowerCase();
      return l.includes("drown") || l.includes("emerg");
    }).length;
  }, [alerts]);

  // Cancel speech immediately when muted or bell turned off
  useEffect(() => {
    if (!soundEnabled || !autoAnnounce) {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    }
  }, [soundEnabled, autoAnnounce]);

  // Voice alert for emergencies on all zones - ONLY when bell (autoAnnounce) is ON AND sound is enabled
  useEmergencyVoiceAlert(autoAnnounce && soundEnabled ? alerts.items : null, soundEnabled && autoAnnounce && !paused, openZone, paused);
  
  // Zone-specific alerts when a zone modal is open (plays when zone is open AND sound enabled)
  useEmergencyVoiceAlert(openZone && soundEnabled ? modalAlerts.items : null, soundEnabled && !paused && !modalPaused && !!openZone, openZone, paused || modalPaused);

  // Announce zone status when opening a zone (only once per zone open)
  // This always works when a zone is opened, regardless of bell toggle
  const lastAnnouncedZoneRef = useRef(null);
  const zoneAnnouncedRef = useRef(false);
  useEffect(() => {
    if (!openZone || paused || !soundEnabled) {
      lastAnnouncedZoneRef.current = null;
      zoneAnnouncedRef.current = false;
      return;
    }
    
    // Only announce once when zone opens (not on re-renders or data changes)
    if (lastAnnouncedZoneRef.current === openZone && zoneAnnouncedRef.current) return;
    lastAnnouncedZoneRef.current = openZone;
    
    // Wait a bit for detections data to load, then announce once
    const timer = setTimeout(() => {
      if (zoneAnnouncedRef.current) return; // Double check
      if (!soundEnabled) return; // Re-check in case muted during timeout
      zoneAnnouncedRef.current = true;
      
      const emergencyAlerts = (modalAlerts.items || []).filter((a) => {
        const l = String(a.label || "").toLowerCase();
        return l.includes("drown") || l.includes("emerg");
      });
      
      let message;
      if (emergencyAlerts.length > 0) {
        message = `Alert! Drowning detected in Zone ${openZone}. Check immediately.`;
      } else {
        message = `Zone ${openZone} monitoring active. No emergency detected.`;
      }
      
      speakAnnouncement(message, 1.0);
    }, 1200);
    
    return () => clearTimeout(timer);
  }, [openZone, paused, soundEnabled]); // Zone open announcement respects soundEnabled

  // Auto announce all zones periodically when enabled
  const lastAutoAnnounceRef = useRef(0);
  useEffect(() => {
    if (!autoAnnounce || !soundEnabled || paused) return;
    
    const announceStatus = () => {
      const now = Date.now();
      if (now - lastAutoAnnounceRef.current < 45000) return; // At least 45 seconds between auto announcements
      lastAutoAnnounceRef.current = now;
      
      const allAlerts = alerts.items || [];
      const drowningAlerts = allAlerts.filter(a => {
        const l = String(a.label || "").toLowerCase();
        return l.includes("drown") || l.includes("emerg");
      });
      
      // Get zones with drowning
      const affectedZones = [...new Set(drowningAlerts.map(a => a.zone || "unknown"))];
      
      let message;
      if (affectedZones.length === 0) {
        message = `All zones clear and safe. No drowning detected.`;
      } else if (affectedZones.length === 1) {
        message = `Alert! Drowning detected in Zone ${affectedZones[0]}. Check immediately.`;
      } else if (affectedZones.length === 2) {
        message = `Alert! Drowning detected in Zone ${affectedZones[0]} and Zone ${affectedZones[1]}. Check immediately.`;
      } else {
        // For 3+ zones: "Zone 1, Zone 2, and Zone 3"
        const lastZone = affectedZones[affectedZones.length - 1];
        const otherZones = affectedZones.slice(0, -1).map(z => `Zone ${z}`).join(", ");
        message = `Alert! Drowning detected in ${otherZones}, and Zone ${lastZone}. Check immediately.`;
      }
      
      speakAnnouncement(message, 1.0);
    };
    
    // Announce immediately when turned on
    announceStatus();
    
    // Then check every 45 seconds
    const interval = setInterval(announceStatus, 45000);
    
    return () => clearInterval(interval);
  }, [autoAnnounce, soundEnabled, paused, alerts.items, zones.length]);

  // Fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  }, []);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, []);

  // Export alerts to CSV
  const exportToCSV = useCallback(() => {
    const items = alerts.items || [];
    if (items.length === 0) return;
    const headers = ["Timestamp", "Zone", "Label", "Confidence"];
    const rows = items.map(a => [
      a.timestamp || "",
      a.zone || "",
      a.label || "",
      a.confidence ? (a.confidence * 100).toFixed(1) + "%" : ""
    ]);
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `coastvision_alerts_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [alerts]);

  return (
    <Box sx={{ minHeight: "100vh", background: "linear-gradient(175deg, #0a2e38 0%, #0d3b47 8%, #0f4150 18%, #0c3545 35%, #0a2c3a 55%, #091e2b 75%, #071520 100%)", color: "#e7eefc", pt: 2, position: "relative", overflow: "hidden" }}>
      {/* Animated Ocean Waves Background */}
      <Box sx={{ position: "fixed", bottom: 0, left: 0, right: 0, height: "220px", zIndex: 0, pointerEvents: "none", opacity: 0.15 }}>
        <svg viewBox="0 0 1440 220" preserveAspectRatio="none" style={{ width: "100%", height: "100%", display: "block" }}>
          <defs>
            <linearGradient id="waveGrad1" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#0d9488" stopOpacity="0.1" />
            </linearGradient>
            <linearGradient id="waveGrad2" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#5eead4" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#14b8a6" stopOpacity="0.05" />
            </linearGradient>
          </defs>
          <path fill="url(#waveGrad1)" d="M0,120 C360,180 720,60 1080,120 C1260,150 1380,90 1440,120 L1440,220 L0,220 Z">
            <animate attributeName="d" dur="8s" repeatCount="indefinite" values="M0,120 C360,180 720,60 1080,120 C1260,150 1380,90 1440,120 L1440,220 L0,220 Z;M0,140 C360,80 720,160 1080,100 C1260,80 1380,140 1440,100 L1440,220 L0,220 Z;M0,120 C360,180 720,60 1080,120 C1260,150 1380,90 1440,120 L1440,220 L0,220 Z" />
          </path>
          <path fill="url(#waveGrad2)" d="M0,160 C480,120 960,200 1440,140 L1440,220 L0,220 Z">
            <animate attributeName="d" dur="6s" repeatCount="indefinite" values="M0,160 C480,120 960,200 1440,140 L1440,220 L0,220 Z;M0,140 C480,200 960,120 1440,170 L1440,220 L0,220 Z;M0,160 C480,120 960,200 1440,140 L1440,220 L0,220 Z" />
          </path>
        </svg>
      </Box>
      {/* Subtle top shimmer */}
      <Box sx={{ position: "fixed", top: 0, left: 0, right: 0, height: "300px", zIndex: 0, pointerEvents: "none", background: "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(45,212,191,0.06) 0%, transparent 70%)" }} />
      {/* NAVBAR - Premium dark chrome header with gold accents */}
      <AppBar position="fixed" elevation={0} sx={{ top: 0, left: 0, right: 0, zIndex: 1100, background: "linear-gradient(180deg, rgba(8,26,36,0.97) 0%, rgba(6,20,28,0.97) 100%)", backdropFilter: "blur(30px) saturate(200%)", WebkitBackdropFilter: "blur(30px) saturate(200%)", borderBottom: "none", borderRadius: { xs: 0, md: "0 0 16px 16px" }, boxShadow: "0 12px 50px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.05), inset 0 -1px 0 rgba(0,0,0,0.3)", border: "1px solid rgba(45,212,191,0.08)", "&::after": { content: '""', position: "absolute", bottom: 0, left: "5%", right: "5%", height: "1px", background: "linear-gradient(90deg, transparent, rgba(45,212,191,0.25), transparent)" } }}>
        <Toolbar sx={{ gap: 3, minHeight: 110, px: { xs: 2, md: 5 }, py: 2 }}>
          {/* Logo - Premium */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 2.5 }}>
            <Box sx={{ position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
              {/* Outer ocean accent ring */}
              <Box sx={{ position: "absolute", inset: -5, borderRadius: "16px", background: "linear-gradient(135deg, rgba(45,212,191,0.4) 0%, rgba(20,184,166,0.15) 50%, rgba(45,212,191,0.4) 100%)", filter: "blur(4px)", animation: "oceanPulse 4s ease-in-out infinite", "@keyframes oceanPulse": { "0%, 100%": { opacity: 0.5 }, "50%": { opacity: 0.9 } } }} />
              {/* Logo shield / emblem */}
              <Box sx={{ position: "relative", zIndex: 1, width: 62, height: 62, borderRadius: "14px", background: "linear-gradient(145deg, #0a2e38 0%, #071520 100%)", border: "2px solid rgba(45,212,191,0.4)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 4px 24px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06)" }}>
                <WavesIcon sx={{ fontSize: 32, color: "#2dd4bf", filter: "drop-shadow(0 0 6px rgba(45,212,191,0.5))" }} />
              </Box>
              {/* Status dot */}
              <Box sx={{ position: "absolute", bottom: -2, right: -2, width: 14, height: 14, borderRadius: "50%", bgcolor: "#00e676", border: "2.5px solid #16161a", boxShadow: "0 0 8px rgba(0,230,118,0.6)", zIndex: 2 }} />
            </Box>
            <Box>
              <Box sx={{ display: "flex", alignItems: "baseline", gap: 0.3 }}>
                <Typography sx={{ fontWeight: 300, fontSize: 34, letterSpacing: 3, lineHeight: 1, color: "rgba(255,255,255,0.92)", fontFamily: "'Inter', 'Segoe UI', sans-serif" }}>COAST</Typography>
                <Typography sx={{ fontWeight: 800, fontSize: 34, letterSpacing: 1, lineHeight: 1, background: "linear-gradient(135deg, #5eead4 0%, #2dd4bf 40%, #14b8a6 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>VISION</Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                <Box sx={{ width: 20, height: 1, bgcolor: "rgba(45,212,191,0.4)" }} />
                <Typography sx={{ fontSize: 10, color: "rgba(45,212,191,0.7)", fontWeight: 600, letterSpacing: 5, textTransform: "uppercase" }}>AI Surveillance</Typography>
                <Box sx={{ width: 20, height: 1, bgcolor: "rgba(45,212,191,0.4)" }} />
              </Box>
            </Box>
          </Box>

          <Divider orientation="vertical" flexItem sx={{ mx: 3, borderColor: "rgba(255,255,255,0.08)", height: 55, alignSelf: "center" }} />

          {/* Status chips */}
          <Stack direction="row" spacing={2}>
            <Tooltip title={backendOk ? "Backend connected" : "Backend offline"}>
              <Chip
                icon={<FiberManualRecordIcon sx={{ fontSize: 12 }} />}
                label={backendOk === false ? "Offline" : backendOk === true ? "Online" : "…"}
                sx={{ bgcolor: backendOk === false ? "rgba(244,67,54,0.15)" : "rgba(0,230,118,0.1)", color: backendOk === false ? "#ff6b6b" : "#00e676", fontWeight: 700, fontSize: 13, height: 40, px: 1.5, border: `1.5px solid ${backendOk === false ? "rgba(244,67,54,0.3)" : "rgba(0,230,118,0.3)"}`, borderRadius: "10px", "& .MuiChip-icon": { color: backendOk === false ? "#ff6b6b" : "#00e676" } }}
              />
            </Tooltip>
            <Tooltip title={health?.gpu_name || "Processing device"}>
              <Chip
                icon={<VideocamIcon sx={{ fontSize: 18 }} />}
                label={health?.device ?? "?"}
                sx={{ bgcolor: "rgba(45,212,191,0.1)", color: "#2dd4bf", fontWeight: 700, fontSize: 13, height: 40, px: 1.5, border: "1.5px solid rgba(45,212,191,0.3)", borderRadius: "10px", "& .MuiChip-icon": { color: "#2dd4bf" } }}
              />
            </Tooltip>
            <Chip
              label={`${zones.length} Zones`}
              sx={{ bgcolor: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.75)", fontWeight: 700, fontSize: 13, height: 40, px: 1.5, border: "1.5px solid rgba(255,255,255,0.12)", borderRadius: "10px" }}
            />
          </Stack>

          <Box sx={{ flex: 1 }} />

          {/* Alert badge */}
          <Tooltip title={`${analysis.alerts_total ?? 0} total alerts`}>
            <Badge badgeContent={emergencyCount} color="error" max={99} sx={{ "& .MuiBadge-badge": { bgcolor: "#ff1744", boxShadow: "0 0 12px rgba(255,23,68,0.6)", fontSize: 12, fontWeight: 800 } }}>
              <Chip
                icon={<NotificationsActiveIcon sx={{ fontSize: 20 }} />}
                label={`${analysis.alerts_total ?? 0} Alerts`}
                sx={{ bgcolor: emergencyCount > 0 ? "rgba(255,23,68,0.15)" : "rgba(255,171,0,0.1)", color: emergencyCount > 0 ? "#ff6b6b" : "#ffc107", fontWeight: 700, fontSize: 13, height: 40, px: 1.5, border: `1.5px solid ${emergencyCount > 0 ? "rgba(255,23,68,0.35)" : "rgba(255,193,7,0.3)"}`, borderRadius: "10px", "& .MuiChip-icon": { color: emergencyCount > 0 ? "#ff6b6b" : "#ffc107" } }}
              />
            </Badge>
          </Tooltip>

          <Divider orientation="vertical" flexItem sx={{ mx: 3, borderColor: "rgba(255,255,255,0.08)", height: 55, alignSelf: "center" }} />

          {/* Actions */}
          <Tooltip title={soundEnabled ? "Mute all voice alerts" : "Enable voice alerts"}>
            <IconButton
              onClick={() => {
                setSoundEnabled(s => {
                  if (s && 'speechSynthesis' in window) window.speechSynthesis.cancel();
                  return !s;
                });
              }}
              sx={{ color: soundEnabled ? "#00e676" : "rgba(255,255,255,0.35)", bgcolor: soundEnabled ? "rgba(0,230,118,0.1)" : "rgba(255,255,255,0.06)", border: `1.5px solid ${soundEnabled ? "rgba(0,230,118,0.3)" : "rgba(255,255,255,0.1)"}`, borderRadius: "12px", width: 46, height: 46, "&:hover": { color: soundEnabled ? "#00e676" : "#2dd4bf", bgcolor: soundEnabled ? "rgba(0,230,118,0.18)" : "rgba(45,212,191,0.1)", borderColor: soundEnabled ? "rgba(0,230,118,0.5)" : "rgba(45,212,191,0.3)" } }}
            >
              {soundEnabled ? <VolumeUpIcon sx={{ fontSize: 22 }} /> : <VolumeOffIcon sx={{ fontSize: 22 }} />}
            </IconButton>
          </Tooltip>

          <Tooltip title={isFullscreen ? "Exit fullscreen" : "Fullscreen mode"}>
            <IconButton
              onClick={toggleFullscreen}
              sx={{ color: "rgba(255,255,255,0.5)", bgcolor: "rgba(255,255,255,0.06)", border: "1.5px solid rgba(255,255,255,0.1)", borderRadius: "12px", width: 46, height: 46, "&:hover": { color: "#2dd4bf", bgcolor: "rgba(45,212,191,0.1)", borderColor: "rgba(45,212,191,0.3)" } }}
            >
              {isFullscreen ? <FullscreenExitIcon sx={{ fontSize: 22 }} /> : <FullscreenIcon sx={{ fontSize: 22 }} />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Export alerts to CSV">
            <IconButton
              onClick={exportToCSV}
              sx={{ color: "rgba(255,255,255,0.5)", bgcolor: "rgba(255,255,255,0.06)", border: "1.5px solid rgba(255,255,255,0.1)", borderRadius: "12px", width: 46, height: 46, "&:hover": { color: "#2dd4bf", bgcolor: "rgba(45,212,191,0.1)", borderColor: "rgba(45,212,191,0.3)" } }}
            >
              <DownloadIcon sx={{ fontSize: 22 }} />
            </IconButton>
          </Tooltip>

          <Tooltip title={autoAnnounce ? "Turn off auto announcements" : "Turn on auto announcements"}>
            <IconButton
              onClick={() => {
                setAutoAnnounce(prev => {
                  if (prev && 'speechSynthesis' in window) window.speechSynthesis.cancel();
                  return !prev;
                });
              }}
              sx={{ 
                color: autoAnnounce ? (emergencyCount > 0 ? "#ff6b6b" : "#00e676") : "rgba(255,255,255,0.35)", 
                bgcolor: autoAnnounce ? (emergencyCount > 0 ? "rgba(255,107,107,0.12)" : "rgba(0,230,118,0.1)") : "rgba(255,255,255,0.06)", 
                border: `1.5px solid ${autoAnnounce ? (emergencyCount > 0 ? "rgba(255,107,107,0.3)" : "rgba(0,230,118,0.3)") : "rgba(255,255,255,0.1)"}`, 
                borderRadius: "12px",
                width: 46, 
                height: 46, 
                animation: autoAnnounce && emergencyCount > 0 ? "pulse 1.5s infinite" : "none",
                "&:hover": { 
                  color: autoAnnounce ? (emergencyCount > 0 ? "#ff6b6b" : "#00e676") : "#2dd4bf", 
                  bgcolor: autoAnnounce ? (emergencyCount > 0 ? "rgba(255,107,107,0.2)" : "rgba(0,230,118,0.18)") : "rgba(45,212,191,0.1)", 
                  borderColor: autoAnnounce ? (emergencyCount > 0 ? "rgba(255,107,107,0.5)" : "rgba(0,230,118,0.5)") : "rgba(45,212,191,0.3)" 
                } 
              }}
            >
              <NotificationsActiveIcon sx={{ fontSize: 22 }} />
            </IconButton>
          </Tooltip>

          <Tooltip title="Reload zones">
            <IconButton
              onClick={() => fetch(`${API}/api/zones/reload`, { method: "POST" }).catch(() => {})}
              sx={{ color: "rgba(255,255,255,0.5)", bgcolor: "rgba(255,255,255,0.06)", border: "1.5px solid rgba(255,255,255,0.1)", borderRadius: "12px", width: 46, height: 46, "&:hover": { color: "#2dd4bf", bgcolor: "rgba(45,212,191,0.1)", borderColor: "rgba(45,212,191,0.3)" } }}
            >
              <RefreshIcon sx={{ fontSize: 22 }} />
            </IconButton>
          </Tooltip>

          <Button
            variant={paused ? "outlined" : "contained"}
            startIcon={paused ? <PlayArrowIcon /> : <PauseIcon />}
            onClick={() => {
              setPaused((p) => {
                if (!p && 'speechSynthesis' in window) {
                  window.speechSynthesis.cancel();
                }
                return !p;
              });
            }}
            sx={{ background: paused ? "transparent" : "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)", borderColor: "#2dd4bf", borderWidth: 2, color: paused ? "#2dd4bf" : "#071520", fontWeight: 800, fontSize: 14, textTransform: "none", px: 4, py: 1.2, borderRadius: "12px", boxShadow: paused ? "none" : "0 4px 24px rgba(45,212,191,0.35)", "&:hover": { borderWidth: 2, background: paused ? "rgba(45,212,191,0.12)" : "linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)" } }}
          >
            {paused ? "Resume" : "Pause All"}
          </Button>
        </Toolbar>

        {/* Tabs - Sleek dark strip with gold accent */}
        <Box sx={{ background: "rgba(255,255,255,0.02)", borderRadius: { xs: 0, md: "0 0 14px 14px" }, borderTop: "1px solid rgba(255,255,255,0.04)" }}>
          <Tabs
            value={tab}
            onChange={(_, v) => setTab(v)}
            textColor="inherit"
            TabIndicatorProps={{ style: { background: "linear-gradient(90deg, transparent, #2dd4bf, transparent)", height: 2.5, borderRadius: 2 } }}
            sx={{ px: { xs: 2, md: 5 }, minHeight: 52, "& .MuiTab-root": { minHeight: 52, textTransform: "none", fontWeight: 600, fontSize: 13.5, color: "rgba(255,255,255,0.35)", letterSpacing: 0.5, gap: 1.5, px: 3, mx: 0.5, my: 0.5, borderRadius: "10px", transition: "all 0.25s ease", "&:hover": { color: "rgba(255,255,255,0.75)", bgcolor: "rgba(255,255,255,0.04)" }, "&.Mui-selected": { color: "#5eead4", bgcolor: "rgba(45,212,191,0.08)", fontWeight: 700 } } }}
          >
            <Tab icon={<DashboardIcon sx={{ fontSize: 19 }} />} iconPosition="start" label="Dashboard" />
            <Tab icon={<AnalyticsIcon sx={{ fontSize: 19 }} />} iconPosition="start" label="Analytics" />
            <Tab icon={<HistoryIcon sx={{ fontSize: 19 }} />} iconPosition="start" label="Event Logs" />
            <Tab icon={<SettingsIcon sx={{ fontSize: 19 }} />} iconPosition="start" label="Settings" />
            <Tab icon={<SecurityIcon sx={{ fontSize: 19 }} />} iconPosition="start" label="Lifeguards" />
            <Tab icon={<VideoLibraryIcon sx={{ fontSize: 19 }} />} iconPosition="start" label="Videos" />
          </Tabs>
        </Box>
      </AppBar>

      {/* CONTENT AREA */}
      <Box sx={{ p: { xs: 2, md: 5 }, pt: { xs: "200px", md: "210px" }, maxWidth: 2000, mx: "auto", minHeight: "calc(100vh - 180px)", position: "relative", zIndex: 1 }}>
        {tab === 0 && (
          <Box>
            {/* Stats Row */}
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "repeat(2, 1fr)", md: "repeat(4, 1fr)" }, gap: 3, mb: 4 }}>
              {[
                { label: "Active Zones", value: zones.length, icon: <VideocamIcon />, color: "#00e5ff" },
                { label: "Total Alerts", value: analysis.alerts_total ?? 0, icon: <NotificationsActiveIcon />, color: "#ffab00" },
                { label: "Emergencies", value: emergencyCount, icon: <WarningAmberIcon />, color: "#ff5252" },
                { label: "Device", value: health?.device ?? "—", icon: <SettingsIcon />, color: "#00e676" },
              ].map((stat) => (
                <Box
                  key={stat.label}
                  sx={{ p: 3.5, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", gap: 3, transition: "all 0.3s", boxShadow: "0 4px 24px rgba(0,0,0,0.3)", "&:hover": { bgcolor: "#131f2b", borderColor: `${stat.color}40`, transform: "translateY(-3px)", boxShadow: `0 12px 40px ${stat.color}15` } }}
                >
                  <Avatar sx={{ bgcolor: `${stat.color}20`, color: stat.color, width: 60, height: 60, boxShadow: `0 0 25px ${stat.color}30` }}>{React.cloneElement(stat.icon, { sx: { fontSize: 30 } })}</Avatar>
                  <Box>
                    <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.5)", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1.5 }}>{stat.label}</Typography>
                    <Typography sx={{ fontSize: 32, fontWeight: 900, mt: 0.5, color: "#fff" }}>{stat.value}</Typography>
                  </Box>
                </Box>
              ))}
            </Box>

            {/* Section Title */}
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 4, mt: 2 }}>
              <Box>
                <Typography sx={{ fontWeight: 900, fontSize: 26, color: "#fff", display: "flex", alignItems: "center", gap: 2 }}>
                  <VideocamIcon sx={{ color: "#2dd4bf", fontSize: 32 }} />
                  Live Zone Feeds
                </Typography>
                <Typography sx={{ fontSize: 15, color: "rgba(255,255,255,0.5)", mt: 0.5, ml: 6 }}>Real-time surveillance monitoring across all beach zones</Typography>
              </Box>
              <Chip label={`${zones.length} Active`} sx={{ bgcolor: "rgba(45,212,191,0.12)", color: "#2dd4bf", fontWeight: 800, fontSize: 14, height: 36, px: 1, border: "2px solid rgba(45,212,191,0.35)" }} />
            </Box>

            {/* Zone Grid */}
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", lg: "repeat(3, 1fr)", xl: "repeat(3, 1fr)" }, gap: 4 }}>
            {zones.length === 0 ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 3, alignItems: "center", justifyContent: "center", textAlign: "center", gridColumn: "1/-1", py: 14, borderRadius: 4, bgcolor: "#0f1923", border: "2px dashed rgba(255,255,255,0.1)" }}>
                <Avatar sx={{ bgcolor: "rgba(45,212,191,0.15)", width: 90, height: 90, boxShadow: "0 0 40px rgba(45,212,191,0.2)" }}>
                  <VideocamIcon sx={{ fontSize: 45, color: "#2dd4bf" }} />
                </Avatar>
                <Box>
                  <Typography sx={{ fontWeight: 900, fontSize: 26, color: "#fff" }}>No Zones Detected</Typography>
                  <Typography sx={{ color: "rgba(255,255,255,0.5)", maxWidth: 480, mt: 1.5, fontSize: 16 }}>Drop any video file into the Videos tab to get started, or add files to the videos folder and click reload.</Typography>
                </Box>
                <Button startIcon={<RefreshIcon />} variant="contained" onClick={() => fetch(`${API}/api/zones/reload`, { method: "POST" })} sx={{ mt: 2, background: "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)", color: "#071520", fontWeight: 800, px: 5, py: 1.5, fontSize: 15, boxShadow: "0 4px 30px rgba(45,212,191,0.35)", "&:hover": { background: "linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)" } }}>Reload Zones</Button>
              </Box>
            ) : (
              zones.map((z) => {
                const meta = zoneMeta.get(z) || {};
                const exists = !!meta.exists;
                const showError = typeof meta.last_error === "string" && meta.last_error.length > 0;

                return (
                  <Card
                    key={z}
                    elevation={0}
                    onClick={() => { setModalBlobUrl(`${API}/api/zones/${z}/frame.jpg?t=${Date.now()}`); setOpenZone(z); }}
                    sx={{
                      cursor: "pointer",
                      bgcolor: "#0f1923",
                      border: "1px solid rgba(255,255,255,0.06)",
                      borderRadius: 4,
                      overflow: "hidden",
                      transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                      boxShadow: "0 4px 30px rgba(0,0,0,0.3)",
                      "&:hover": { borderColor: "rgba(255,255,255,0.12)", transform: "translateY(-4px)", boxShadow: "0 20px 60px rgba(0,0,0,0.4)" },
                    }}
                  >
                    {/* Zone Header with Label */}
                    <Box sx={{ p: 2.5, borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", justifyContent: "space-between", bgcolor: "rgba(255,255,255,0.02)" }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                        <Avatar sx={{ background: "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)", width: 48, height: 48, fontSize: 18, fontWeight: 900, color: "#071520", boxShadow: "0 4px 20px rgba(45,212,191,0.3)" }}>{z}</Avatar>
                        <Box>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            <ZoneNameEditor
                              zoneId={z}
                              currentName={zoneNames.get(z) || `Zone ${z}`}
                              api={API}
                              onNameChanged={(newName) => {
                                setZoneNames(prev => new Map(prev.set(z, newName)));
                              }}
                            />
                          </Box>
                          <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.4)", fontWeight: 600, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={meta.filename || ""}>{exists ? (meta.filename || "Live Feed") : "No Video Source"}</Typography>
                        </Box>
                      </Box>
                      <Chip
                        icon={<FiberManualRecordIcon sx={{ fontSize: 8, animation: exists ? "pulse 1.5s infinite" : "none", "@keyframes pulse": { "0%, 100%": { opacity: 1 }, "50%": { opacity: 0.4 } } }} />}
                        label={exists ? "LIVE" : "OFFLINE"}
                        size="small"
                        sx={{ bgcolor: exists ? "rgba(105,240,174,0.15)" : "rgba(255,82,82,0.15)", color: exists ? "#69f0ae" : "#ff5252", fontWeight: 800, fontSize: 10, border: `1px solid ${exists ? "rgba(105,240,174,0.3)" : "rgba(255,82,82,0.3)"}`, "& .MuiChip-icon": { color: exists ? "#69f0ae" : "#ff5252" } }}
                      />
                    </Box>

                    {/* Video Feed Area */}
                    <Box sx={{ position: "relative", height: 380 }}>
                      <ZoneCardPlayback zid={z} paused={paused} />

                      {(!exists || showError) && (
                        <Box sx={{ position: "absolute", inset: 0, zIndex: 4, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", bgcolor: "rgba(7,21,32,.93)", backdropFilter: "blur(8px)", px: 3, textAlign: "center" }}>
                          <Avatar sx={{ bgcolor: "rgba(255,171,0,0.15)", width: 64, height: 64, mb: 2 }}>
                            <WarningAmberIcon sx={{ fontSize: 32, color: "#ffab00" }} />
                          </Avatar>
                          <Typography sx={{ fontWeight: 900, fontSize: 18, color: "#fff" }}>{exists ? "Stream Error" : "Video Missing"}</Typography>
                          <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.5)", mt: 1 }}>{showError ? meta.last_error : "No video source configured for this zone"}</Typography>
                        </Box>
                      )}
                    </Box>
                  </Card>
                );
              })
            )}
            </Box>
          </Box>
        )}

        {tab === 1 && (
          <Box>
            {/* Header */}
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
              <Box>
                <Typography sx={{ fontWeight: 900, fontSize: 28, color: "#fff", display: "flex", alignItems: "center", gap: 2 }}>
                  <Box sx={{ p: 1.5, borderRadius: 2, background: "linear-gradient(135deg, rgba(45,212,191,0.2) 0%, rgba(45,212,191,0.05) 100%)", display: "flex", border: "1px solid rgba(45,212,191,0.15)" }}>
                    <AnalyticsIcon sx={{ color: "#2dd4bf", fontSize: 28 }} />
                  </Box>
                  Analytics Dashboard
                </Typography>
                <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 14, mt: 1, ml: 7 }}>Real-time surveillance insights</Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 2, py: 1, borderRadius: 2, bgcolor: "rgba(52,211,153,0.08)", border: "1px solid rgba(52,211,153,0.2)" }}>
                <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: "#34d399", animation: "pulse 1.5s infinite" }} />
                <Typography sx={{ fontSize: 12, color: "#34d399", fontWeight: 600 }}>LIVE</Typography>
              </Box>
            </Box>

            {/* Key Metrics Row — always visible */}
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "repeat(2, 1fr)", lg: "repeat(4, 1fr)" }, gap: 3, mb: 3 }}>
              {[
                { title: "Total Detections", value: analysis.alerts_total ?? 0, icon: "📊", gradient: "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)", accent: "#2dd4bf" },
                { title: "Monitored Zones", value: zones.length, icon: "🎯", gradient: "linear-gradient(135deg, #34d399 0%, #10b981 100%)", accent: "#34d399" },
                { title: "Emergency Alerts", value: emergencyCount, icon: "🚨", gradient: "linear-gradient(135deg, #ff5252 0%, #cc4141 100%)", accent: "#ff5252" },
                { title: "Avg Confidence", value: `${((alerts.items?.reduce((a, b) => a + (b.conf || 0), 0) / Math.max(1, alerts.items?.length || 1)) * 100).toFixed(0)}%`, icon: "⚡", gradient: "linear-gradient(135deg, #ffab00 0%, #ff8f00 100%)", accent: "#ffab00" },
              ].map((card) => (
                <Box key={card.title} sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
                    <Box sx={{ p: 1.5, borderRadius: 2, background: card.gradient, boxShadow: "0 4px 15px rgba(0,0,0,0.3)" }}>
                      <Typography sx={{ fontSize: 24 }}>{card.icon}</Typography>
                    </Box>
                  </Box>
                  <Typography sx={{ fontSize: 48, fontWeight: 900, color: "#fff", lineHeight: 1, letterSpacing: "-1px" }}>{card.value}</Typography>
                  <Typography sx={{ fontSize: 15, color: "rgba(255,255,255,0.5)", fontWeight: 600, mt: 1.5 }}>{card.title}</Typography>
                </Box>
              ))}
            </Box>

            {/* ── Analytics Sub-Section Tabs ── */}
            <Box sx={{ 
              display: "flex", gap: 1, mb: 3, p: 0.75, borderRadius: 3, bgcolor: "#0a1018", 
              border: "1px solid rgba(255,255,255,0.06)", overflowX: "auto",
              "&::-webkit-scrollbar": { height: 0 }
            }}>
              {[
                { key: "overview", label: "Overview", icon: <DashboardIcon sx={{ fontSize: 18 }} /> },
                { key: "timeline", label: "Person Count", icon: <TrendingUpIcon sx={{ fontSize: 18 }} /> },
                { key: "crowd", label: "Crowd Density", icon: <PersonIcon sx={{ fontSize: 18 }} /> },
                { key: "detections", label: "Detections", icon: <NotificationsActiveIcon sx={{ fontSize: 18 }} /> },
                { key: "activity", label: "Live Feed", icon: <HistoryIcon sx={{ fontSize: 18 }} /> },
                { key: "response_times", label: "Response Times", icon: <AccessTimeIcon sx={{ fontSize: 18 }} /> },
              ].map((s) => (
                <Button
                  key={s.key}
                  startIcon={s.icon}
                  onClick={() => setAnalyticsSection(s.key)}
                  sx={{
                    flex: { xs: 1, md: "none" },
                    px: 3, py: 1.2,
                    borderRadius: 2.5,
                    fontWeight: 800,
                    fontSize: 13,
                    textTransform: "none",
                    whiteSpace: "nowrap",
                    color: analyticsSection === s.key ? "#071520" : "rgba(255,255,255,0.55)",
                    bgcolor: analyticsSection === s.key ? "#2dd4bf" : "transparent",
                    boxShadow: analyticsSection === s.key ? "0 4px 20px rgba(45,212,191,0.35)" : "none",
                    transition: "all 0.25s cubic-bezier(.4,0,.2,1)",
                    "&:hover": {
                      bgcolor: analyticsSection === s.key ? "#14b8a6" : "rgba(255,255,255,0.06)",
                    },
                  }}
                >
                  {s.label}
                </Button>
              ))}
            </Box>

            {/* ── OVERVIEW SECTION ── */}
            {analyticsSection === "overview" && (
              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "420px 1fr" }, gap: 4 }}>
                
                {/* Pie Chart - Detection Types */}
                <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 4 }}>
                    <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>Detection Types</Typography>
                    <Chip label="Distribution" size="small" sx={{ bgcolor: "rgba(45,212,191,0.1)", color: "#2dd4bf", fontWeight: 600, fontSize: 12, height: 26 }} />
                  </Box>
                  
                  {Object.keys(analysis.alerts_by_label || {}).length > 0 ? (
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
                      <Box sx={{ position: "relative", width: 240, height: 240 }}>
                        <svg width="240" height="240" viewBox="0 0 100 100">
                          <defs>
                            <filter id="pieGlow" x="-50%" y="-50%" width="200%" height="200%">
                              <feGaussianBlur stdDeviation="1.5" result="glow"/>
                              <feMerge><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
                            </filter>
                          </defs>
                          <circle cx="50" cy="50" r="40" fill="transparent" stroke="rgba(255,255,255,0.03)" strokeWidth="16" />
                          {(() => {
                            const entries = Object.entries(analysis.alerts_by_label || {});
                            const total = entries.reduce((a, [, v]) => a + v, 0);
                            const colors = ["#2dd4bf", "#ff5252", "#34d399", "#f59e0b", "#a78bfa"];
                            let cumulative = 0;
                            return entries.map(([label, count], i) => {
                              const pct = (count / total) * 100;
                              const offset = cumulative;
                              cumulative += pct;
                              return (
                                <circle key={label} cx="50" cy="50" r="40" fill="transparent" stroke={colors[i % colors.length]} strokeWidth="16" strokeDasharray={`${pct * 2.51} ${251 - pct * 2.51}`} strokeDashoffset={-offset * 2.51 + 62.75} filter="url(#pieGlow)" style={{ transition: "all 0.5s ease" }} />
                              );
                            });
                          })()}
                          <circle cx="50" cy="50" r="30" fill="#0f1923" />
                        </svg>
                        <Box sx={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                          <Typography sx={{ fontSize: 42, fontWeight: 900, color: "#fff", lineHeight: 1, letterSpacing: "-1px" }}>{analysis.alerts_total ?? 0}</Typography>
                          <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.45)", fontWeight: 700, mt: 0.5, letterSpacing: "2px" }}>TOTAL</Typography>
                        </Box>
                      </Box>
                      
                      <Stack spacing={1} sx={{ width: "100%" }}>
                        {(() => {
                          const entries = Object.entries(analysis.alerts_by_label || {});
                          const total = entries.reduce((a, [, v]) => a + v, 0);
                          const colors = ["#2dd4bf", "#ff5252", "#34d399", "#f59e0b", "#a78bfa"];
                          return entries.map(([label, count], i) => {
                            const pct = ((count / total) * 100).toFixed(0);
                            const isEmergency = label.toLowerCase().includes("drown") || label.toLowerCase().includes("emerg");
                            return (
                              <Box key={label} sx={{ display: "flex", alignItems: "center", gap: 2, p: 2, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)", transition: "all 0.2s", "&:hover": { bgcolor: "rgba(255,255,255,0.04)" } }}>
                                <Box sx={{ width: 12, height: 12, borderRadius: 1, bgcolor: colors[i % colors.length] }} />
                                <Typography sx={{ flex: 1, fontSize: 15, fontWeight: 600, color: "rgba(255,255,255,0.8)", textTransform: "capitalize" }}>{label}</Typography>
                                <Typography sx={{ fontSize: 18, fontWeight: 800, color: colors[i % colors.length] }}>{count}</Typography>
                                <Typography sx={{ fontSize: 14, color: "rgba(255,255,255,0.5)", minWidth: 40, fontWeight: 600 }}>{pct}%</Typography>
                                {isEmergency && <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "#ff5252", animation: "pulse 1s infinite" }} />}
                              </Box>
                            );
                          });
                        })()}
                      </Stack>
                    </Box>
                  ) : (
                    <Box sx={{ py: 6, textAlign: "center" }}>
                      <Box sx={{ width: 60, height: 60, borderRadius: "50%", bgcolor: "rgba(255,255,255,0.03)", display: "flex", alignItems: "center", justifyContent: "center", mx: "auto", mb: 2 }}>
                        <AnalyticsIcon sx={{ fontSize: 28, color: "rgba(255,255,255,0.2)" }} />
                      </Box>
                      <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 13 }}>No detection data</Typography>
                    </Box>
                  )}
                </Box>

                {/* Bar Chart - Zone Activity */}
                <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 4 }}>
                    <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>Zone Activity</Typography>
                    <Chip label={`${zones.length} Zones`} size="small" sx={{ bgcolor: "rgba(45,212,191,0.1)", color: "#2dd4bf", fontWeight: 600, fontSize: 12, height: 26 }} />
                  </Box>
                  
                  {zones.length > 0 ? (
                    <Box>
                      <Box sx={{ display: "flex", gap: 2, height: 320 }}>
                        <Box sx={{ display: "flex", flexDirection: "column", justifyContent: "space-between", py: 1, pr: 1 }}>
                          {(() => {
                            const maxCount = Math.max(...Object.values(analysis.alerts_by_zone || { default: 0 }), 1);
                            return [maxCount, Math.round(maxCount * 0.75), Math.round(maxCount * 0.5), Math.round(maxCount * 0.25), 0].map((val) => (
                              <Typography key={val} sx={{ fontSize: 13, color: "rgba(255,255,255,0.4)", minWidth: 28, textAlign: "right", fontWeight: 600 }}>{val}</Typography>
                            ));
                          })()}
                        </Box>
                        <Box sx={{ flex: 1, position: "relative", borderLeft: "1px solid rgba(255,255,255,0.08)", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                          <Box sx={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", justifyContent: "space-between", pointerEvents: "none" }}>
                            {[0, 1, 2, 3].map((i) => (
                              <Box key={i} sx={{ height: 1, bgcolor: "rgba(255,255,255,0.04)", width: "100%" }} />
                            ))}
                          </Box>
                          <Box sx={{ display: "flex", alignItems: "flex-end", height: "100%", gap: 2, px: 2, pb: 0.5 }}>
                            {(() => {
                              const maxCount = Math.max(...Object.values(analysis.alerts_by_zone || { default: 1 }), 1);
                              const colors = ["#2dd4bf", "#34d399", "#f59e0b", "#a78bfa", "#f472b6", "#22d3ee"];
                              return zones.map((zone, idx) => {
                                const count = (analysis.alerts_by_zone || {})[zone] || 0;
                                const height = (count / maxCount) * 100;
                                const color = colors[idx % colors.length];
                                return (
                                  <Box key={zone} sx={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" }}>
                                    <Box sx={{ flex: 1, width: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end" }}>
                                      {count > 0 && <Typography sx={{ fontSize: 14, fontWeight: 800, color: color, mb: 0.5 }}>{count}</Typography>}
                                      <Box sx={{ width: "75%", maxWidth: 56, height: `${height}%`, minHeight: count > 0 ? 12 : 3, background: `linear-gradient(180deg, ${color} 0%, ${color}70 100%)`, borderRadius: "6px 6px 0 0", boxShadow: count > 0 ? `0 0 20px ${color}30` : "none", transition: "height 0.5s ease", position: "relative", "&::before": count > 0 ? { content: '""', position: "absolute", top: 0, left: 0, right: 0, height: "40%", background: "linear-gradient(180deg, rgba(255,255,255,0.25) 0%, transparent 100%)", borderRadius: "6px 6px 0 0" } : {} }} />
                                    </Box>
                                  </Box>
                                );
                              });
                            })()}
                          </Box>
                        </Box>
                      </Box>
                      <Box sx={{ display: "flex", pl: 4, mt: 1.5 }}>
                        {zones.map((zone, idx) => {
                          const colors = ["#2dd4bf", "#34d399", "#f59e0b", "#a78bfa", "#f472b6", "#22d3ee"];
                          return (
                            <Box key={zone} sx={{ flex: 1, textAlign: "center" }}>
                              <Typography sx={{ fontSize: 14, color: colors[idx % colors.length], fontWeight: 800 }}>{zoneNames.get(zone) || `Zone ${zone}`}</Typography>
                            </Box>
                          );
                        })}
                      </Box>
                      <Box sx={{ display: "flex", gap: 2, mt: 3, pt: 3, borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                        {[
                          { label: "Most Active", value: Object.entries(analysis.alerts_by_zone || {}).sort((a, b) => b[1] - a[1])[0]?.[0] ? (zoneNames.get(Number(Object.entries(analysis.alerts_by_zone || {}).sort((a, b) => b[1] - a[1])[0]?.[0])) || `Zone ${Object.entries(analysis.alerts_by_zone || {}).sort((a, b) => b[1] - a[1])[0]?.[0]}`) : "\u2014", color: "#2dd4bf" },
                          { label: "Avg/Zone", value: zones.length > 0 ? ((analysis.alerts_total || 0) / zones.length).toFixed(1) : "0", color: "#34d399" },
                          { label: "Coverage", value: zones.length > 0 ? `${((Object.keys(analysis.alerts_by_zone || {}).length / zones.length) * 100).toFixed(0)}%` : "0%", color: "#ffab00" },
                        ].map((stat) => (
                          <Box key={stat.label} sx={{ flex: 1, p: 2.5, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", textAlign: "center" }}>
                            <Typography sx={{ fontSize: 24, fontWeight: 900, color: stat.color, letterSpacing: "-0.5px" }}>{stat.value}</Typography>
                            <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.5)", mt: 0.5, fontWeight: 600 }}>{stat.label}</Typography>
                          </Box>
                        ))}
                      </Box>
                    </Box>
                  ) : (
                    <Box sx={{ height: 260, display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Box sx={{ textAlign: "center" }}>
                        <VideocamIcon sx={{ fontSize: 40, color: "rgba(255,255,255,0.15)", mb: 1 }} />
                        <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 13 }}>No zones configured</Typography>
                      </Box>
                    </Box>
                  )}
                </Box>
              </Box>
            )}

            {/* ── PERSON COUNT TIMELINE SECTION ── */}
            {analyticsSection === "timeline" && (
              <Box>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                    <TrendingUpIcon sx={{ color: "#2dd4bf", fontSize: 24 }} />
                    <Box>
                      <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>Person Count Timeline</Typography>
                      <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 13 }}>Real-time person count trends across all zones</Typography>
                    </Box>
                  </Box>
                  <Chip label="Live Updates" sx={{ bgcolor: "rgba(45,212,191,0.12)", color: "#2dd4bf", fontWeight: 800, fontSize: 12, height: 28 }} />
                </Box>
                
                {zones.length > 0 ? (
                  <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "repeat(2, 1fr)" }, gap: 3 }}>
                    {zones.map((zoneId) => (
                      <PersonCountTimeline
                        key={zoneId}
                        zoneId={zoneId}
                        zoneName={zoneNames.get(zoneId) || `Zone ${zoneId}`}
                        api={API}
                      />
                    ))}
                  </Box>
                ) : (
                  <Box sx={{ p: 8, textAlign: "center", borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <TrendingUpIcon sx={{ fontSize: 48, color: "rgba(255,255,255,0.1)", mb: 2 }} />
                    <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 16 }}>No zones available</Typography>
                  </Box>
                )}
              </Box>
            )}

            {/* ── CROWD DENSITY ANALYTICS SECTION ── */}
            {analyticsSection === "crowd" && (
              <CrowdDensityAnalytics api={API} zones={zones} zoneNames={zoneNames} />
            )}

            {/* ── DETECTIONS SECTION ── */}
            {analyticsSection === "detections" && (
              <Box>
                <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
                    <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>Detection Moments</Typography>
                    <Chip label="Timeline" size="small" sx={{ bgcolor: "rgba(45,212,191,0.1)", color: "#2dd4bf", fontWeight: 600, fontSize: 12, height: 26 }} />
                  </Box>
                  {(alerts.items || []).length > 0 ? (
                    <Box>
                      {/* Confidence bar chart */}
                      <Box sx={{ display: "flex", alignItems: "flex-end", gap: "3px", height: 160, mb: 2, px: 1 }}>
                        {(() => {
                          const recent = (alerts.items || []).slice(0, 40).reverse();
                          return recent.map((item, idx) => {
                            const isEmergency = String(item.label || "").toLowerCase().includes("drown") || String(item.label || "").toLowerCase().includes("emerg");
                            const color = isEmergency ? "#ff5252" : "#2dd4bf";
                            const conf = (item.conf || 0.5);
                            const h = Math.max(15, conf * 100);
                            return (
                              <Tooltip key={idx} title={`${item.label} (${(conf * 100).toFixed(0)}%) - Zone ${item.zone}`} arrow>
                                <Box sx={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", height: "100%" }}>
                                  <Box sx={{ width: "100%", height: `${h}%`, minHeight: 8, maxWidth: 20, bgcolor: color, borderRadius: "3px 3px 0 0", opacity: 0.5 + conf * 0.5, transition: "all 0.25s", cursor: "pointer", "&:hover": { opacity: 1, transform: "scaleY(1.08)" } }} />
                                </Box>
                              </Tooltip>
                            );
                          });
                        })()}
                      </Box>
                      <Box sx={{ display: "flex", justifyContent: "space-between", px: 1, mb: 3 }}>
                        <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.3)", fontWeight: 600 }}>Earliest</Typography>
                        <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.3)", fontWeight: 600 }}>Most Recent</Typography>
                      </Box>
                      
                      {/* Stats row */}
                      <Box sx={{ display: "flex", gap: 2, pt: 3, borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                        {(() => {
                          const items = alerts.items || [];
                          const drowningCount = items.filter(i => String(i.label || "").toLowerCase().includes("drown")).length;
                          const normalCount = items.length - drowningCount;
                          return [
                            { label: "Normal Detections", value: normalCount, color: "#2dd4bf", icon: "\uD83D\uDC41" },
                            { label: "Emergency Events", value: drowningCount, color: "#ff5252", icon: "\uD83D\uDEA8" },
                            { label: "Total Events", value: items.length, color: "#34d399", icon: "\uD83D\uDCCA" },
                          ].map(s => (
                            <Box key={s.label} sx={{ flex: 1, p: 2.5, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", textAlign: "center", border: "1px solid rgba(255,255,255,0.04)" }}>
                              <Typography sx={{ fontSize: 16, mb: 0.5 }}>{s.icon}</Typography>
                              <Typography sx={{ fontSize: 28, fontWeight: 900, color: s.color, letterSpacing: "-0.5px" }}>{s.value}</Typography>
                              <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.4)", mt: 0.5, fontWeight: 600 }}>{s.label}</Typography>
                            </Box>
                          ));
                        })()}
                      </Box>
                    </Box>
                  ) : (
                    <Box sx={{ py: 6, textAlign: "center" }}>
                      <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 14 }}>No detection moments recorded yet</Typography>
                    </Box>
                  )}
                </Box>

                {/* Per-Zone Detection Breakdown */}
                {zones.length > 0 && Object.keys(analysis.alerts_by_zone || {}).length > 0 && (
                  <Box sx={{ mt: 3, display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)", xl: "repeat(3, 1fr)" }, gap: 3 }}>
                    {zones.map((zoneId) => {
                      const count = (analysis.alerts_by_zone || {})[zoneId] || 0;
                      const colors = ["#2dd4bf", "#34d399", "#f59e0b", "#a78bfa", "#f472b6", "#22d3ee"];
                      const color = colors[(zoneId - 1) % colors.length];
                      const total = analysis.alerts_total || 1;
                      const pct = ((count / total) * 100).toFixed(1);
                      return (
                        <Box key={zoneId} sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)" }}>
                          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
                            <Typography sx={{ fontWeight: 800, fontSize: 16, color: "#fff" }}>{zoneNames.get(zoneId) || `Zone ${zoneId}`}</Typography>
                            <Chip label={`${pct}%`} size="small" sx={{ bgcolor: `${color}20`, color: color, fontWeight: 800, fontSize: 12 }} />
                          </Box>
                          <Box sx={{ height: 6, borderRadius: 3, bgcolor: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                            <Box sx={{ height: "100%", width: `${Math.min(100, (count / Math.max(...Object.values(analysis.alerts_by_zone || { d: 1 }), 1)) * 100)}%`, bgcolor: color, borderRadius: 3, transition: "width 0.5s ease" }} />
                          </Box>
                          <Typography sx={{ fontSize: 32, fontWeight: 900, color: color, mt: 1.5 }}>{count}</Typography>
                          <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.4)" }}>detections</Typography>
                        </Box>
                      );
                    })}
                  </Box>
                )}
              </Box>
            )}

            {/* ── LIVE ACTIVITY FEED SECTION ── */}
            {analyticsSection === "activity" && (
              <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                    <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>Recent Activity</Typography>
                    <Chip label={`${(alerts.items || []).length} Events`} size="small" sx={{ bgcolor: "rgba(255,255,255,0.05)", color: "rgba(255,255,255,0.6)", fontWeight: 600, fontSize: 12, height: 24 }} />
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "#34d399", animation: "pulse 1.5s infinite" }} />
                    <Typography sx={{ fontSize: 11, color: "#34d399", fontWeight: 600 }}>LIVE</Typography>
                  </Box>
                </Box>
                
                <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)", xl: "repeat(3, 1fr)" }, gap: 2, maxHeight: 600, overflowY: "auto", pr: 1, "&::-webkit-scrollbar": { width: 4 }, "&::-webkit-scrollbar-thumb": { bgcolor: "rgba(255,255,255,0.1)", borderRadius: 2 } }}>
                  {(alerts.items || []).length > 0 ? (
                    (alerts.items || []).slice(0, 30).map((alert, idx) => {
                      const isEmergency = String(alert.label || "").toLowerCase().includes("drown") || String(alert.label || "").toLowerCase().includes("emerg");
                      const color = isEmergency ? "#ff5252" : "#2dd4bf";
                      return (
                        <Box key={idx} sx={{ display: "flex", alignItems: "center", gap: 2, p: 2, borderRadius: 2, bgcolor: isEmergency ? "rgba(255,82,82,0.06)" : "rgba(45,212,191,0.04)", border: `1px solid ${isEmergency ? "rgba(255,82,82,0.15)" : "rgba(45,212,191,0.08)"}`, transition: "all 0.2s", "&:hover": { bgcolor: isEmergency ? "rgba(255,82,82,0.1)" : "rgba(45,212,191,0.08)" } }}>
                          <Box sx={{ width: 3, height: 36, borderRadius: 1, bgcolor: color, flexShrink: 0 }} />
                          <Box sx={{ flex: 1, minWidth: 0 }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                              <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#fff", textTransform: "capitalize" }}>{alert.label || "Detection"}</Typography>
                              <Chip label={zoneNames.get(alert.zone) || `Z${alert.zone}`} size="small" sx={{ height: 16, fontSize: 9, fontWeight: 700, bgcolor: "rgba(45,212,191,0.1)", color: "#2dd4bf" }} />
                              {isEmergency && <Box sx={{ width: 5, height: 5, borderRadius: "50%", bgcolor: "#ff5252", animation: "pulse 0.8s infinite" }} />}
                            </Box>
                            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                              <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.35)" }}>{alert.ts}</Typography>
                              <Typography sx={{ fontSize: 11, fontWeight: 700, color: color }}>{((alert.conf || 0) * 100).toFixed(0)}%</Typography>
                            </Box>
                          </Box>
                        </Box>
                      );
                    })
                  ) : (
                    <Box sx={{ gridColumn: "1 / -1", py: 6, textAlign: "center" }}>
                      <HistoryIcon sx={{ fontSize: 36, color: "rgba(255,255,255,0.1)", mb: 1 }} />
                      <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 13 }}>Waiting for activity...</Typography>
                    </Box>
                  )}
                </Box>
              </Box>
            )}

            {/* ── RESPONSE TIMES ANALYTICS SECTION ── */}
            {analyticsSection === "response_times" && (
              <ResponseTimeAnalytics api={API} />
            )}
          </Box>
        )}

        {tab === 2 && (
          <Box>
            <Typography sx={{ fontWeight: 800, fontSize: 20, mb: 3 }}>Event History</Typography>
            <Box sx={{ borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", overflow: "hidden", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
              {/* Header */}
              <Box sx={{ display: "grid", gridTemplateColumns: "180px 100px 140px 1fr 80px", gap: 2, p: 2, bgcolor: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                <Typography sx={{ fontSize: 12, fontWeight: 700, color: "rgba(255,255,255,0.5)", textTransform: "uppercase" }}>Timestamp</Typography>
                <Typography sx={{ fontSize: 12, fontWeight: 700, color: "rgba(255,255,255,0.5)", textTransform: "uppercase" }}>Zone</Typography>
                <Typography sx={{ fontSize: 12, fontWeight: 700, color: "rgba(255,255,255,0.5)", textTransform: "uppercase" }}>Type</Typography>
                <Typography sx={{ fontSize: 12, fontWeight: 700, color: "rgba(255,255,255,0.5)", textTransform: "uppercase" }}>Message</Typography>
                <Typography sx={{ fontSize: 12, fontWeight: 700, color: "rgba(255,255,255,0.5)", textTransform: "uppercase", textAlign: "right" }}>Conf</Typography>
              </Box>
              {/* Rows */}
              <Stack spacing={0}>
                {(alerts.items || []).map((a, idx) => {
                  const isEmergency = String(a.label || "").toLowerCase().includes("drown") || String(a.label || "").toLowerCase().includes("emerg");
                  return (
                    <Box key={idx} sx={{ display: "grid", gridTemplateColumns: "180px 100px 140px 1fr 80px", gap: 2, p: 2, borderBottom: "1px solid rgba(255,255,255,0.05)", bgcolor: isEmergency ? "rgba(244,67,54,0.08)" : "transparent", "&:hover": { bgcolor: "rgba(255,255,255,0.03)" } }}>
                      <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.6)" }}>{a.ts}</Typography>
                      <Chip label={`Zone ${a.zone}`} size="small" sx={{ width: "fit-content", bgcolor: "rgba(0,188,212,0.15)", color: "#00bcd4", fontWeight: 700, fontSize: 11, height: 22 }} />
                      <Typography sx={{ fontSize: 13, fontWeight: 700, color: isEmergency ? "#f44336" : "#ff9800", textTransform: "capitalize" }}>{a.label}</Typography>
                      <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.8)" }}>{a.msg}</Typography>
                      <Typography sx={{ fontSize: 13, textAlign: "right", fontWeight: 700 }}>{((a.conf ?? 0) * 100).toFixed(0)}%</Typography>
                    </Box>
                  );
                })}
                {(alerts.items || []).length === 0 && <Typography sx={{ p: 4, textAlign: "center", color: "rgba(255,255,255,0.4)" }}>No events recorded yet</Typography>}
              </Stack>
            </Box>
          </Box>
        )}

        {tab === 3 && (
          <Box>
            <Typography sx={{ fontWeight: 800, fontSize: 20, mb: 3 }}>System Settings</Typography>

            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" }, gap: 3 }}>
              <Box sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
                <Typography sx={{ fontWeight: 700, mb: 2, color: "#2dd4bf" }}>Backend Configuration</Typography>
                <Stack spacing={1.5}>
                  {[
                    { label: "API Endpoint", value: API },
                    { label: "Status", value: backendOk ? "Connected" : "Offline" },
                    { label: "Device", value: health?.device },
                    { label: "GPU", value: health?.gpu_name || "N/A" },
                    { label: "VRAM", value: health?.gpu_vram_gb ? `${health.gpu_vram_gb} GB` : "N/A" },
                  ].map((item) => (
                    <Box key={item.label} sx={{ display: "flex", justifyContent: "space-between", py: 1, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                      <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 13 }}>{item.label}</Typography>
                      <Typography sx={{ fontWeight: 700, fontSize: 13 }}>{item.value || "—"}</Typography>
                    </Box>
                  ))}
                </Stack>
              </Box>

              <Box sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
                <Typography sx={{ fontWeight: 700, mb: 2, color: "#f59e0b" }}>Detection Settings</Typography>
                <Stack spacing={1.5}>
                  {[
                    { label: "Python Version", value: health?.python_version },
                    { label: "Platform", value: health?.platform?.split("-")[0] },
                    { label: "FPS", value: health?.fps },
                    { label: "Confidence Threshold", value: health?.conf },
                    { label: "Alert Confidence", value: health?.alert_conf },
                    { label: "IOU Threshold", value: health?.iou },
                  ].map((item) => (
                    <Box key={item.label} sx={{ display: "flex", justifyContent: "space-between", py: 1, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                      <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 13 }}>{item.label}</Typography>
                      <Typography sx={{ fontWeight: 700, fontSize: 13 }}>{item.value ?? "—"}</Typography>
                    </Box>
                  ))}
                </Stack>
              </Box>

              {/* Voice Alert Settings */}
              <Box sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)", gridColumn: { md: "span 2" } }}>
                <Typography sx={{ fontWeight: 700, mb: 2, color: "#34d399" }}>Voice Alert Settings</Typography>
                <Stack spacing={2}>
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", py: 1, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                    <Box>
                      <Typography sx={{ fontSize: 14, fontWeight: 600 }}>Emergency Voice Alerts</Typography>
                      <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>Play alarm and voice announcement when drowning is detected</Typography>
                    </Box>
                    <Button
                      variant={soundEnabled ? "contained" : "outlined"}
                      onClick={() => {
                        setSoundEnabled(s => {
                          if (s && 'speechSynthesis' in window) window.speechSynthesis.cancel();
                          return !s;
                        });
                      }}
                      startIcon={soundEnabled ? <VolumeUpIcon /> : <VolumeOffIcon />}
                      sx={{ 
                        background: soundEnabled ? "linear-gradient(135deg, #34d399 0%, #10b981 100%)" : "transparent",
                        borderColor: "#34d399",
                        color: soundEnabled ? "#071520" : "#34d399",
                        fontWeight: 700,
                        textTransform: "none",
                        "&:hover": { background: soundEnabled ? "linear-gradient(135deg, #10b981 0%, #059669 100%)" : "rgba(52,211,153,0.15)" }
                      }}
                    >
                      {soundEnabled ? "Enabled" : "Disabled"}
                    </Button>
                  </Box>
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", py: 1 }}>
                    <Box>
                      <Typography sx={{ fontSize: 14, fontWeight: 600 }}>Test Voice Alert</Typography>
                      <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>Click to hear a sample drowning detection announcement</Typography>
                    </Box>
                    <Button
                      variant="outlined"
                      onClick={() => {
                        // Play test alarm (shorter)
                        try {
                          const ctx = new (window.AudioContext || window.webkitAudioContext)();
                          for (let i = 0; i < 2; i++) {
                            const osc = ctx.createOscillator();
                            const gain = ctx.createGain();
                            osc.connect(gain);
                            gain.connect(ctx.destination);
                            osc.frequency.setValueAtTime(800, ctx.currentTime + i * 0.25);
                            osc.frequency.setValueAtTime(600, ctx.currentTime + i * 0.25 + 0.1);
                            gain.gain.setValueAtTime(0.3, ctx.currentTime + i * 0.25);
                            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + i * 0.25 + 0.2);
                            osc.start(ctx.currentTime + i * 0.25);
                            osc.stop(ctx.currentTime + i * 0.25 + 0.2);
                          }
                        } catch (e) {}
                        // Speak test message with male voice
                        if ('speechSynthesis' in window) {
                          window.speechSynthesis.cancel();
                          const utterance = new SpeechSynthesisUtterance("Alert! Drowning detected in Zone 1. Please check immediately.");
                          utterance.rate = 1.0;
                          utterance.volume = 1.0;
                          utterance.pitch = 0.9;
                          utterance.lang = 'en-US';
                          const maleV = getMaleVoice();
                          if (maleV) utterance.voice = maleV;
                          setTimeout(() => window.speechSynthesis.speak(utterance), 400);
                        }
                      }}
                      sx={{ 
                        borderColor: "#ffab00",
                        color: "#ffab00",
                        fontWeight: 700,
                        textTransform: "none",
                        "&:hover": { bgcolor: "rgba(255,171,0,0.15)", borderColor: "#ffab00" }
                      }}
                    >
                      Test Alert
                    </Button>
                  </Box>
                </Stack>
              </Box>

              {/* Lifeguard Readiness */}
              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1.1fr 1.1fr" }, gap: 3 }}>
                <Box sx={{ p: 3, borderRadius: 3, bgcolor: "#020617", border: "1px solid rgba(148,163,184,0.35)", boxShadow: "0 18px 45px rgba(15,23,42,0.95)", position: "relative", overflow: "hidden" }}>
                  <Box
                    sx={{
                      position: "absolute",
                      inset: -40,
                      background: "radial-gradient(circle at top left, rgba(45,212,191,0.25), transparent 60%)",
                      opacity: 0.8,
                      pointerEvents: "none",
                    }}
                  />
                  <Stack spacing={2.5} sx={{ position: "relative" }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                      <Avatar sx={{ bgcolor: "rgba(56,189,248,0.14)", border: "1px solid rgba(56,189,248,0.4)", width: 42, height: 42 }}>
                        <SecurityIcon sx={{ fontSize: 24, color: "#38bdf8" }} />
                      </Avatar>
                      <Box>
                        <Typography sx={{ fontSize: 15, fontWeight: 700, letterSpacing: 0.4 }}>Lifeguard Readiness Overview</Typography>
                        <Typography sx={{ fontSize: 12, color: "rgba(148,163,184,0.9)" }}>
                          Keep at least one lifeguard registered for every active zone to ensure no area is unprotected.
                        </Typography>
                      </Box>
                    </Box>

                    <Divider sx={{ borderColor: "rgba(148,163,184,0.18)" }} />

                    <Stack spacing={1.5}>
                      <Typography sx={{ fontSize: 12, color: "rgba(148,163,184,0.9)", textTransform: "uppercase", letterSpacing: 1.5 }}>
                        Quick Tips
                      </Typography>
                      <Stack spacing={1} sx={{ fontSize: 12.5, color: "rgba(226,232,240,0.9)" }}>
                        <Box sx={{ display: "flex", gap: 1 }}>
                          <CheckCircleIcon sx={{ fontSize: 16, color: "#4ade80", mt: "2px" }} />
                          <Typography sx={{ fontSize: 12.5 }}>
                            Create each lifeguard account in the Lifeguards tab with name and phone number.
                          </Typography>
                        </Box>
                        <Box sx={{ display: "flex", gap: 1 }}>
                          <CheckCircleIcon sx={{ fontSize: 16, color: "#f97316", mt: "2px" }} />
                          <Typography sx={{ fontSize: 12.5 }}>
                            Assign zones using the zone chips so lifeguards only see their beach sections in the mobile app.
                          </Typography>
                        </Box>
                        <Box sx={{ display: "flex", gap: 1 }}>
                          <CheckCircleIcon sx={{ fontSize: 16, color: "#38bdf8", mt: "2px" }} />
                          <Typography sx={{ fontSize: 12.5 }}>
                            Lifeguards install Expo Go, connect to the same Wi-Fi, and sign in with their phone number.
                          </Typography>
                        </Box>
                      </Stack>
                    </Stack>

                    <Divider sx={{ borderColor: "rgba(148,163,184,0.18)" }} />

                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mt: 1 }}>
                      <Box>
                        <Typography sx={{ fontSize: 12, color: "rgba(148,163,184,0.9)" }}>Status summary</Typography>
                        <Typography sx={{ fontSize: 13, fontWeight: 600 }}>
                          Manage app accounts in the Lifeguards tab.
                        </Typography>
                      </Box>
                    </Box>
                  </Stack>
                </Box>
              </Box>
            </Box>
          </Box>
        )}

        {tab === 4 && (
          <Box>
            <Typography sx={{ fontWeight: 800, fontSize: 20, mb: 3 }}>Lifeguard Accounts</Typography>
            <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.45)", mb: 3 }}>
              Create mobile app accounts and assign zones. Lifeguards sign in on the CoastVision mobile app with their phone number.
            </Typography>
            <LifeguardAccountsPanel api={API} />
          </Box>
        )}

        {tab === 5 && (
          <VideoManager api={API} onReload={() => fetch(`${API}/api/zones/reload`, { method: "POST" }).catch(() => {})} />
        )}
      </Box>

      <Dialog
        open={!!openZone}
        onClose={() => setOpenZone(null)}
        maxWidth="xl"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: "#0a2e38",
            borderRadius: 4,
            border: "1px solid rgba(45,212,191,0.15)",
            p: 0,
            overflow: "hidden",
            width: { xs: "100vw", md: "95vw" },
            height: { xs: "100vh", md: "92vh" },
            maxWidth: "none",
            m: { xs: 0, md: 2 },
            boxShadow: "0 25px 80px rgba(0,0,0,0.6), 0 0 80px rgba(45,212,191,0.06)",
          },
        }}
      >
        <Box sx={{ color: "#fff", height: "100%", display: "flex", flexDirection: "column" }}>
          {/* Enhanced Modal Header */}
          <Box sx={{ 
            p: 2.5, 
            display: "flex", 
            alignItems: "center", 
            justifyContent: "space-between", 
            background: "linear-gradient(180deg, rgba(45,212,191,0.06) 0%, transparent 100%)",
            borderBottom: "1px solid rgba(45,212,191,0.1)"
          }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
              <Box sx={{ position: "relative" }}>
                <Avatar sx={{ 
                  background: "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)", 
                  fontWeight: 900, 
                  width: 56, 
                  height: 56,
                  fontSize: 24,
                  color: "#071520",
                  boxShadow: "0 0 30px rgba(45,212,191,0.35)",
                  border: "3px solid rgba(45,212,191,0.3)"
                }}>{openZone}</Avatar>
                <Box sx={{ 
                  position: "absolute", 
                  bottom: 2, 
                  right: 2, 
                  width: 14, 
                  height: 14, 
                  borderRadius: "50%", 
                  bgcolor: modalPaused ? "#ffab00" : "#34d399", 
                  border: "2px solid #0a2e38",
                  boxShadow: `0 0 10px ${modalPaused ? "rgba(255,171,0,0.6)" : "rgba(52,211,153,0.6)"}`
                }} />
              </Box>
              <Box>
                <Typography sx={{ fontWeight: 900, fontSize: 26, background: "linear-gradient(135deg, #fff 0%, #2dd4bf 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Zone {openZone}</Typography>
                {(() => { const m = zoneMeta.get(openZone); return m?.filename ? <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.35)", fontFamily: "monospace", mt: -0.5 }}>{m.filename}</Typography> : null; })()}
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 0.5 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <MyLocationIcon sx={{ fontSize: 14, color: "#2dd4bf" }} />
                    <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.6)" }}>Live Monitoring</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <SpeedIcon sx={{ fontSize: 14, color: "#34d399" }} />
                    <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.6)" }}>Real-time AI</Typography>
                  </Box>
                </Stack>
              </Box>
            </Box>
            
            <Stack direction="row" spacing={1.5} alignItems="center">
              <Chip 
                icon={<AccessTimeIcon sx={{ fontSize: 16 }} />}
                label={modalDetections.age_s != null ? `${modalDetections.age_s.toFixed(1)}s ago` : "Live"}
                sx={{ bgcolor: "rgba(52,211,153,0.1)", color: "#34d399", fontWeight: 700, fontSize: 12, border: "1px solid rgba(52,211,153,0.25)", "& .MuiChip-icon": { color: "#34d399" } }}
              />
              <Chip 
                icon={<CenterFocusStrongIcon sx={{ fontSize: 16 }} />}
                label={`${modalDetections.count ?? 0} Detected`}
                sx={{ bgcolor: "rgba(45,212,191,0.1)", color: "#2dd4bf", fontWeight: 700, fontSize: 12, border: "1px solid rgba(45,212,191,0.25)", "& .MuiChip-icon": { color: "#2dd4bf" } }}
              />
              <Box sx={{ width: 1, height: 32, bgcolor: "rgba(255,255,255,0.1)", mx: 1 }} />
              
              {/* Sound Toggle */}
              <Tooltip title={soundEnabled ? "Mute voice alerts" : "Enable voice alerts"}>
                <IconButton 
                  onClick={() => setSoundEnabled(s => !s)}
                  sx={{ 
                    color: soundEnabled ? "#34d399" : "rgba(255,255,255,0.4)", 
                    bgcolor: soundEnabled ? "rgba(52,211,153,0.12)" : "rgba(255,255,255,0.1)",
                    border: `1px solid ${soundEnabled ? "rgba(52,211,153,0.25)" : "rgba(255,255,255,0.2)"}`,
                    "&:hover": { bgcolor: soundEnabled ? "rgba(52,211,153,0.2)" : "rgba(255,255,255,0.15)" }
                  }}
                >
                  {soundEnabled ? <VolumeUpIcon /> : <VolumeOffIcon />}
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Zoom hint: Scroll to zoom, drag to pan">
                <IconButton sx={{ color: "rgba(255,255,255,0.5)", "&:hover": { color: "#2dd4bf" } }}>
                  <ZoomInIcon />
                </IconButton>
              </Tooltip>
              <Button
                variant={modalPaused ? "outlined" : "contained"}
                startIcon={modalPaused ? <PlayArrowIcon sx={{ fontSize: 28 }} /> : <PauseIcon sx={{ fontSize: 28 }} />}
                onClick={() => {
                  setModalPaused((p) => {
                    if (!p && 'speechSynthesis' in window) {
                      window.speechSynthesis.cancel();
                    }
                    return !p;
                  });
                }}
                sx={{ 
                  background: modalPaused ? "transparent" : "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)", 
                  borderColor: "#2dd4bf", 
                  borderWidth: 2,
                  color: modalPaused ? "#2dd4bf" : "#071520", 
                  fontWeight: 800, 
                  textTransform: "none",
                  px: 4,
                  py: 1.2,
                  fontSize: 16,
                  minWidth: 140,
                  boxShadow: modalPaused ? "none" : "0 4px 20px rgba(45,212,191,0.3)",
                  "&:hover": { borderWidth: 2, background: modalPaused ? "rgba(45,212,191,0.12)" : "linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)" }
                }}
              >
                {modalPaused ? "Resume" : "Pause"}
              </Button>
              <IconButton 
                onClick={() => setOpenZone(null)} 
                sx={{ 
                  color: "rgba(255,255,255,0.7)", 
                  bgcolor: "rgba(255,255,255,0.1)", 
                  border: "1px solid rgba(255,255,255,0.2)",
                  "&:hover": { bgcolor: "rgba(255,107,107,0.2)", borderColor: "rgba(255,107,107,0.5)", color: "#ff6b6b" } 
                }}
              >
                <CloseIcon />
              </IconButton>
            </Stack>
          </Box>

          {/* Modal Content */}
          <Box sx={{ p: 2.5, flex: 1, minHeight: 0, display: "grid", gap: 2.5, gridTemplateColumns: { xs: "1fr", lg: "1fr 340px" }, overflow: "hidden" }}>
            {/* Video Area */}
            <Box sx={{ minHeight: 0, minWidth: 0, display: "flex", flexDirection: "column" }}>
              <Box
                ref={modalVideoBoxRef}
                sx={{
                  flex: 1,
                  minHeight: 0,
                  borderRadius: 3,
                  overflow: "hidden",
                  background: "linear-gradient(135deg, #0a2e38 0%, #071520 100%)",
                  border: "2px solid rgba(0,217,255,0.2)",
                  touchAction: "none",
                  position: "relative",
                  boxShadow: "inset 0 0 100px rgba(0,0,0,0.5)",
                }}
              >
                <TransformWrapper 
                  wheel={{ step: 0.2 }} 
                  doubleClick={{ mode: "zoomIn", step: 0.5 }} 
                  panning={{ disabled: false, velocityDisabled: false }} 
                  pinch={{ disabled: false, step: 10 }} 
                  minScale={1} 
                  maxScale={5}
                  centerOnInit 
                  limitToBounds={false}
                >
                  {({ zoomIn, zoomOut, resetTransform }) => (
                    <>
                      <TransformComponent
                        wrapperStyle={{ width: "100%", height: "100%", cursor: "grab", background: "#000" }}
                        contentStyle={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "#000" }}
                      >
                        <img
                          src={openZone ? modalBlobUrl : ""}
                          alt=""
                          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", display: modalBlobUrl ? "block" : "none" }}
                        />
                        <video
                          ref={modalVideoRef}
                          muted
                          autoPlay
                          playsInline
                          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", background: "transparent", display: !modalPaused && modalMode === "hls" && modalHlsPlaying ? "block" : "none" }}
                        />
                        <img
                          src={openZone && modalMode === "mjpeg" && !modalPaused ? `${API}/api/zones/${openZone}/stream.mjpg` : ""}
                          alt=""
                          onLoad={() => setModalMjpegOk(true)}
                          onError={() => { setModalMode("poll"); setModalMjpegOk(false); }}
                          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", display: openZone && modalMode === "mjpeg" && !modalPaused ? "block" : "none" }}
                        />
                      </TransformComponent>
                      {/* Zoom controls overlay */}
                      <Box sx={{ 
                        position: "absolute", 
                        bottom: 80, 
                        right: 16, 
                        display: "flex", 
                        flexDirection: "column", 
                        gap: 1,
                        zIndex: 20 
                      }}>
                        <Tooltip title="Zoom In" placement="left">
                          <IconButton 
                            onClick={() => zoomIn()} 
                            sx={{ bgcolor: "rgba(0,0,0,0.7)", color: "#2dd4bf", "&:hover": { bgcolor: "rgba(45,212,191,0.2)" } }}
                          >
                            <ZoomInIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Zoom Out" placement="left">
                          <IconButton 
                            onClick={() => zoomOut()} 
                            sx={{ bgcolor: "rgba(0,0,0,0.7)", color: "#2dd4bf", "&:hover": { bgcolor: "rgba(45,212,191,0.2)" } }}
                          >
                            <ZoomOutIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Reset Zoom" placement="left">
                          <IconButton 
                            onClick={() => resetTransform()} 
                            sx={{ bgcolor: "rgba(0,0,0,0.7)", color: "#fff", "&:hover": { bgcolor: "rgba(255,255,255,0.1)" }, fontSize: 12, fontWeight: "bold" }}
                          >
                            1x
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </>
                  )}
                </TransformWrapper>
                
                {/* Video overlay badge */}
                <Box sx={{ 
                  position: "absolute", 
                  top: 16, 
                  left: 16, 
                  px: 2, 
                  py: 0.75, 
                  borderRadius: 2, 
                  bgcolor: "rgba(0,0,0,0.7)", 
                  border: "1px solid rgba(45,212,191,0.2)",
                  backdropFilter: "blur(10px)",
                  display: "flex",
                  alignItems: "center",
                  gap: 1
                }}>
                  <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: modalPaused ? "#ffab00" : "#34d399", animation: modalPaused ? "none" : "pulse 2s infinite" }} />
                  <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#fff" }}>
                    {modalPaused ? "PAUSED" : "LIVE"}
                  </Typography>
                </Box>
                
                {/* Floating bottom control bar */}
                <Box sx={{ 
                  position: "absolute", 
                  bottom: 20, 
                  left: "50%", 
                  transform: "translateX(-50%)",
                  px: 3, 
                  py: 1.5, 
                  borderRadius: 3, 
                  bgcolor: "rgba(0,0,0,0.85)", 
                  border: "1px solid rgba(45,212,191,0.2)",
                  backdropFilter: "blur(15px)",
                  display: "flex",
                  alignItems: "center",
                  gap: 2,
                  boxShadow: "0 8px 30px rgba(0,0,0,0.5)"
                }}>
                  {/* Big Play/Pause button */}
                  <IconButton 
                    onClick={() => {
                      setModalPaused((p) => {
                        if (!p && 'speechSynthesis' in window) {
                          window.speechSynthesis.cancel();
                        }
                        return !p;
                      });
                    }}
                    sx={{ 
                      width: 56, 
                      height: 56, 
                      bgcolor: modalPaused ? "rgba(45,212,191,0.2)" : "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)",
                      background: modalPaused ? "rgba(45,212,191,0.2)" : "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)",
                      color: modalPaused ? "#2dd4bf" : "#071520",
                      border: "2px solid rgba(45,212,191,0.4)",
                      boxShadow: modalPaused ? "none" : "0 0 30px rgba(45,212,191,0.35)",
                      "&:hover": { 
                        bgcolor: modalPaused ? "rgba(45,212,191,0.35)" : "#14b8a6",
                        background: modalPaused ? "rgba(45,212,191,0.35)" : "#14b8a6",
                      }
                    }}
                  >
                    {modalPaused ? <PlayArrowIcon sx={{ fontSize: 36 }} /> : <PauseIcon sx={{ fontSize: 36 }} />}
                  </IconButton>
                  
                  <Box sx={{ width: 1, height: 40, bgcolor: "rgba(255,255,255,0.15)" }} />
                  
                  {/* Detection count */}
                  <Box sx={{ textAlign: "center", minWidth: 80 }}>
                    <Typography sx={{ fontSize: 24, fontWeight: 900, color: "#2dd4bf", lineHeight: 1 }}>{modalDetections.count ?? 0}</Typography>
                    <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.5)", textTransform: "uppercase", letterSpacing: 1 }}>Detected</Typography>
                  </Box>
                  
                  <Box sx={{ width: 1, height: 40, bgcolor: "rgba(255,255,255,0.15)" }} />
                  
                  {/* Sound toggle */}
                  <Tooltip title={soundEnabled ? "Mute" : "Unmute"}>
                    <IconButton 
                      onClick={() => setSoundEnabled(s => !s)}
                      sx={{ 
                        color: soundEnabled ? "#34d399" : "rgba(255,255,255,0.4)", 
                        "&:hover": { color: soundEnabled ? "#34d399" : "#fff" }
                      }}
                    >
                      {soundEnabled ? <VolumeUpIcon sx={{ fontSize: 28 }} /> : <VolumeOffIcon sx={{ fontSize: 28 }} />}
                    </IconButton>
                  </Tooltip>
                  
                  {/* Announce status button */}
                  <Tooltip title="Announce zone status">
                    <IconButton 
                      onClick={() => {
                        const emergencyAlerts = (modalAlerts.items || []).filter((a) => {
                          const l = String(a.label || "").toLowerCase();
                          return l.includes("drown") || l.includes("emerg");
                        });
                        
                        let message;
                        if (emergencyAlerts.length > 0) {
                          message = `Alert! Drowning detected in Zone ${openZone}. Check immediately.`;
                        } else {
                          message = `Zone ${openZone} all clear and safe. No drowning detected.`;
                        }
                        speakAnnouncement(message, 1.0);
                      }}
                      sx={{ 
                        color: "#ffab00", 
                        "&:hover": { color: "#ffc107", bgcolor: "rgba(255,171,0,0.15)" }
                      }}
                    >
                      <NotificationsActiveIcon sx={{ fontSize: 26 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
            </Box>

            {/* Enhanced Sidebar */}
            <Box sx={{ minHeight: 0, display: { xs: "none", lg: "flex" }, flexDirection: "column", gap: 2, overflow: "hidden" }}>
              {/* Live Stats Card */}
              <Box sx={{ 
                p: 2.5, 
                borderRadius: 3, 
                background: "linear-gradient(135deg, rgba(45,212,191,0.12) 0%, rgba(20,184,166,0.06) 100%)",
                border: "1px solid rgba(45,212,191,0.2)",
                position: "relative",
                overflow: "hidden"
              }}>
                <Box sx={{ position: "absolute", top: -30, right: -30, width: 100, height: 100, borderRadius: "50%", bgcolor: "rgba(45,212,191,0.08)" }} />
                <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 800, textTransform: "uppercase", letterSpacing: 1.5, mb: 1 }}>Live Detection</Typography>
                <Box sx={{ display: "flex", alignItems: "baseline", gap: 1 }}>
                  <Typography sx={{ fontSize: 48, fontWeight: 900, color: "#2dd4bf", lineHeight: 1 }}>{modalDetections.count ?? 0}</Typography>
                  <Typography sx={{ fontSize: 16, color: "rgba(255,255,255,0.5)", fontWeight: 600 }}>objects</Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1.5 }}>
                  <TrendingUpIcon sx={{ fontSize: 16, color: "#34d399" }} />
                  <Typography sx={{ fontSize: 12, color: "#34d399", fontWeight: 600 }}>Active monitoring</Typography>
                </Box>
              </Box>
              
              {/* Zone Stats */}
              <Box sx={{ 
                p: 2, 
                borderRadius: 3, 
                bgcolor: "rgba(255,255,255,0.03)", 
                border: "1px solid rgba(255,255,255,0.08)"
              }}>
                <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 800, textTransform: "uppercase", letterSpacing: 1.5, mb: 2 }}>Zone Statistics</Typography>
                <Stack spacing={1.5}>
                  {[
                    { label: "Total Alerts", value: modalAnalysis.alerts_total ?? 0, color: "#ffab00" },
                    { label: "Detection Objects", value: Object.keys(modalAnalysis.alerts_by_label || {}).length, color: "#2dd4bf" },
                    { label: "Stream Status", value: modalMjpegOk ? "Active" : "Polling", color: "#34d399" },
                  ].map(stat => (
                    <Box key={stat.label} sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", py: 1, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                      <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.6)" }}>{stat.label}</Typography>
                      <Typography sx={{ fontSize: 14, fontWeight: 800, color: stat.color }}>{stat.value}</Typography>
                    </Box>
                  ))}
                </Stack>
              </Box>

              {/* Recent Events */}
              <Box sx={{ 
                flex: 1, 
                minHeight: 0, 
                borderRadius: 3, 
                bgcolor: "rgba(255,255,255,0.03)", 
                border: "1px solid rgba(255,255,255,0.08)", 
                p: 2, 
                overflow: "hidden",
                display: "flex",
                flexDirection: "column"
              }}>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
                  <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 800, textTransform: "uppercase", letterSpacing: 1.5 }}>Recent Events</Typography>
                  <Chip label={`${(modalAlerts.items || []).length}`} size="small" sx={{ bgcolor: "rgba(255,255,255,0.1)", height: 20, fontSize: 11, fontWeight: 700 }} />
                </Box>
                <Box sx={{ flex: 1, overflow: "auto", pr: 1, "&::-webkit-scrollbar": { width: 4 }, "&::-webkit-scrollbar-track": { bgcolor: "rgba(255,255,255,0.05)", borderRadius: 2 }, "&::-webkit-scrollbar-thumb": { bgcolor: "rgba(45,212,191,0.25)", borderRadius: 2 } }}>
                  <Stack spacing={1}>
                    {(modalAlerts.items || []).slice(0, 15).map((a, idx) => {
                      const isEmergency = String(a.label || "").toLowerCase().includes("drown") || String(a.label || "").toLowerCase().includes("emerg");
                      return (
                        <Box 
                          key={idx} 
                          sx={{ 
                            p: 1.5, 
                            borderRadius: 2, 
                            bgcolor: isEmergency ? "rgba(255,82,82,0.12)" : "rgba(255,255,255,0.03)", 
                            border: `1px solid ${isEmergency ? "rgba(255,82,82,0.3)" : "rgba(255,255,255,0.06)"}`,
                            transition: "all 0.2s",
                            "&:hover": { bgcolor: isEmergency ? "rgba(255,82,82,0.18)" : "rgba(255,255,255,0.06)" }
                          }}
                        >
                          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                              {isEmergency && <WarningIcon sx={{ fontSize: 14, color: "#ff5252" }} />}
                              <Typography sx={{ fontSize: 13, fontWeight: 700, color: isEmergency ? "#ff5252" : "#fff", textTransform: "capitalize" }}>{a.label}</Typography>
                            </Box>
                            <Chip 
                              label={`${((a.conf ?? 0) * 100).toFixed(0)}%`} 
                              size="small" 
                              sx={{ 
                                bgcolor: isEmergency ? "rgba(255,82,82,0.2)" : "rgba(45,212,191,0.12)", 
                                color: isEmergency ? "#ff5252" : "#2dd4bf", 
                                fontWeight: 800, 
                                fontSize: 10, 
                                height: 18,
                                minWidth: 40
                              }} 
                            />
                          </Box>
                          <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.4)", mt: 0.5 }}>{a.ts}</Typography>
                        </Box>
                      );
                    })}
                    {(modalAlerts.items || []).length === 0 && (
                      <Box sx={{ py: 4, textAlign: "center" }}>
                        <CenterFocusStrongIcon sx={{ fontSize: 32, color: "rgba(255,255,255,0.2)", mb: 1 }} />
                        <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 13 }}>No events detected</Typography>
                      </Box>
                    )}
                  </Stack>
                </Box>
              </Box>
            </Box>
          </Box>
        </Box>
      </Dialog>
    </Box>
  );
}
