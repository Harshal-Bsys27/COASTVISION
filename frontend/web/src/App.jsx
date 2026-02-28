import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
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
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";

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
  // NEW: prefer MJPEG for “real video”; fallback to frame polling if stream fails or paused
  const [useMjpeg, setUseMjpeg] = useState(true);
  const [mjpegOk, setMjpegOk] = useState(false);
  const [blobUrl, setBlobUrl] = useState("");
  const blobUrlRef = useRef("");
  const timerRef = useRef(null);

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

  // MJPEG “loaded” watchdog: if it never loads, switch to fallback
  useEffect(() => {
    setMjpegOk(false);
    if (!useMjpeg || paused) return;
    const t = setTimeout(() => {
      if (!mjpegOk) setUseMjpeg(false);
    }, 1500);
    return () => clearTimeout(t);
  }, [zid, paused, useMjpeg, mjpegOk]);

  // Fallback frame polling -> blob URL (avoids <img> churn + false errors)
  useEffect(() => {
    const shouldPoll = paused || !useMjpeg;
    if (!shouldPoll) {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }

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
    timerRef.current = setInterval(fetchFrame, FALLBACK_FRAME_MS);

    return () => {
      alive = false;
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      ctrl.abort();
    };
  }, [zid, paused, useMjpeg]);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = "";
    };
  }, []);

  const height = "100%";

  return (
    <Box sx={{ position: "relative", width: "100%", height }}>
      {/* Fallback (always works if backend can serve frame.jpg) */}
      <img
        src={blobUrl}
        alt={`Zone ${zid} frame`}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          background: "#050a14",
          userSelect: "none",
          pointerEvents: "none",
          display: paused || !useMjpeg ? "block" : "none",
        }}
      />

      {/* Real “video” playback in cards */}
      <img
        src={!paused && useMjpeg ? `${API}/api/zones/${zid}/stream.mjpg` : ""}
        alt={`Zone ${zid} stream`}
        onLoad={() => setMjpegOk(true)}
        onError={() => {
          setUseMjpeg(false);
          setMjpegOk(false);
        }}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          background: "#050a14",
          userSelect: "none",
          pointerEvents: "none",
          display: !paused && useMjpeg ? "block" : "none",
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
      utterance.lang = 'en-US';
      
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(v => v.name.includes('Google') || v.name.includes('Female'));
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
    utterance.rate = rate; // Slightly slower for clarity
    utterance.volume = 1.0;
    utterance.lang = 'en-US';
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => v.name.includes('Google') || v.name.includes('Female') || v.name.includes('Samantha'));
    if (preferredVoice) utterance.voice = preferredVoice;
    window.speechSynthesis.speak(utterance);
  }
};

export default function App() {
  const [tab, setTab] = useState(0);
  const [paused, setPaused] = useState(false);
  const [openZone, setOpenZone] = useState(null);
  
  // Quick win states
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoAnnounce, setAutoAnnounce] = useState(false); // Auto announce toggle

  const [modalPaused, setModalPaused] = useState(false);
  const [modalUseMjpeg, setModalUseMjpeg] = useState(true);
  const [modalMjpegOk, setModalMjpegOk] = useState(false);

  const [modalBlobUrl, setModalBlobUrl] = useState("");
  const modalBlobUrlRef = useRef("");
  const modalVideoBoxRef = useRef(null);

  const { data: health, ok: backendOk } = usePollJsonWithOk(`${API}/api/health`, 2000, true, {
    status: "unknown",
    device: "?",
    model_names: [],
  });

  const zonesResp = usePollJson(`${API}/api/zones`, 1200, true, { items: [] });
  const zones = useMemo(() => (zonesResp.items || []).map((x) => x.id).filter((x) => Number.isFinite(x)), [zonesResp]);
  const zoneMeta = useMemo(() => new Map((zonesResp.items || []).map((x) => [x.id, x])), [zonesResp]);

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

  // Modal fallback: poll /frame.jpg into a blob URL (always works, avoids “blank”).
  useEffect(() => {
    if (!openZone) return;

    let alive = true;
    let inFlight = false;
    const ctrl = new AbortController();

    const shouldPoll = modalPaused || !modalUseMjpeg || !modalMjpegOk;

    const fetchFrame = async () => {
      if (!alive) return;
      if (!shouldPoll) return;
      if (modalPaused && modalBlobUrlRef.current) return;
      if (inFlight) return;
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
        // ignore transient aborts
      } finally {
        inFlight = false;
      }
    };

    // prime immediately
    fetchFrame();
    const t = shouldPoll && !modalPaused ? setInterval(fetchFrame, MODAL_REFRESH_MS) : null;

    return () => {
      alive = false;
      if (t) clearInterval(t);
      ctrl.abort();
    };
  }, [openZone, modalPaused, modalUseMjpeg, modalMjpegOk]);

  useEffect(() => {
    if (!openZone) {
      setModalPaused(false);
      setModalUseMjpeg(true);
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

  // Voice alert for emergencies - ONLY when bell (autoAnnounce) is ON or a zone is open
  // This prevents alarm sounds when just viewing the dashboard grid
  useEmergencyVoiceAlert(autoAnnounce ? alerts.items : null, soundEnabled && !paused && autoAnnounce, openZone, paused);
  
  // Zone-specific alerts when a zone is open (always plays when zone is open)
  useEmergencyVoiceAlert(openZone ? modalAlerts.items : null, soundEnabled && !paused && !modalPaused && openZone, openZone, paused || modalPaused);

  // Announce zone status when opening a zone (only once per zone open)
  // This always works when a zone is opened, regardless of bell toggle
  const lastAnnouncedZoneRef = useRef(null);
  const zoneAnnouncedRef = useRef(false);
  useEffect(() => {
    if (!openZone || paused) {
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
      zoneAnnouncedRef.current = true;
      
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
    }, 1500);
    
    return () => clearTimeout(timer);
  }, [openZone, paused]); // Zone open announcement always works when zone is opened

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
    <Box sx={{ minHeight: "100vh", bgcolor: "#0a1117", color: "#e7eefc", pt: 2 }}>
      {/* NAVBAR - Premium dark header with glassmorphism */}
      <AppBar position="sticky" elevation={0} sx={{ background: "linear-gradient(90deg, #0d1b2a 0%, #152238 25%, #1a2d47 50%, #152238 75%, #0d1b2a 100%)", borderBottom: "none", borderRadius: { xs: 0, md: "16px" }, mx: { xs: 0, md: 2 }, mt: 1, boxShadow: "0 8px 50px rgba(0,0,0,0.9), 0 4px 25px rgba(0,0,0,0.4)", border: "1px solid rgba(255,255,255,0.08)", "&::before": { content: '""', position: "absolute", top: 0, left: "3%", right: "3%", height: "1px", background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 15%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0.15) 85%, transparent 100%)", borderRadius: "2px" }, "&::after": { content: '""', position: "absolute", bottom: 0, left: "3%", right: "3%", height: "1px", background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.08) 15%, rgba(255,255,255,0.18) 50%, rgba(255,255,255,0.08) 85%, transparent 100%)", borderRadius: "2px" } }}>
        <Toolbar sx={{ gap: 3, minHeight: 110, px: { xs: 2, md: 5 }, py: 2 }}>
          {/* Logo */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
            <Box sx={{ position: "relative" }}>
              <Avatar sx={{ background: "linear-gradient(135deg, #00d9ff 0%, #0096c7 50%, #0077b6 100%)", width: 70, height: 70, boxShadow: "0 0 50px rgba(0,217,255,0.6), 0 0 100px rgba(0,150,199,0.3)", border: "4px solid rgba(0,217,255,0.5)" }}>
                <WavesIcon sx={{ fontSize: 40, filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.5))" }} />
              </Avatar>
              <Box sx={{ position: "absolute", bottom: 2, right: 2, width: 20, height: 20, borderRadius: "50%", bgcolor: "#00ff88", border: "3px solid #1b263b", boxShadow: "0 0 15px rgba(0,255,136,0.8)" }} />
            </Box>
            <Box>
              <Typography sx={{ fontWeight: 900, fontSize: 36, letterSpacing: -0.5, lineHeight: 1.1, background: "linear-gradient(135deg, #ffffff 0%, #00d9ff 50%, #48cae4 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", textShadow: "0 0 40px rgba(0,217,255,0.4)" }}>CoastVision</Typography>
              <Typography sx={{ fontSize: 13, color: "#48cae4", fontWeight: 800, letterSpacing: 5, textTransform: "uppercase", mt: 0.5 }}>AI Beach Surveillance</Typography>
            </Box>
          </Box>

          <Divider orientation="vertical" flexItem sx={{ mx: 3, borderColor: "rgba(0,217,255,0.3)", height: 55, alignSelf: "center" }} />

          {/* Status chips */}
          <Stack direction="row" spacing={2}>
            <Tooltip title={backendOk ? "Backend connected" : "Backend offline"}>
              <Chip
                icon={<FiberManualRecordIcon sx={{ fontSize: 12 }} />}
                label={backendOk === false ? "Offline" : backendOk === true ? "Online" : "…"}
                sx={{ bgcolor: backendOk === false ? "rgba(244,67,54,0.2)" : "rgba(0,255,136,0.12)", color: backendOk === false ? "#ff5252" : "#00ff88", fontWeight: 800, fontSize: 14, height: 42, px: 1.5, border: `2px solid ${backendOk === false ? "rgba(244,67,54,0.4)" : "rgba(0,255,136,0.4)"}`, "& .MuiChip-icon": { color: backendOk === false ? "#ff5252" : "#00ff88" } }}
              />
            </Tooltip>
            <Tooltip title={health?.gpu_name || "Processing device"}>
              <Chip
                icon={<VideocamIcon sx={{ fontSize: 18 }} />}
                label={health?.device ?? "?"}
                sx={{ bgcolor: "rgba(0,217,255,0.12)", color: "#00d9ff", fontWeight: 800, fontSize: 14, height: 42, px: 1.5, border: "2px solid rgba(0,217,255,0.4)", "& .MuiChip-icon": { color: "#00d9ff" } }}
              />
            </Tooltip>
            <Chip
              label={`${zones.length} Zones`}
              sx={{ bgcolor: "rgba(255,255,255,0.1)", color: "#fff", fontWeight: 800, fontSize: 14, height: 42, px: 1.5, border: "2px solid rgba(255,255,255,0.25)" }}
            />
          </Stack>

          <Box sx={{ flex: 1 }} />

          {/* Alert badge */}
          <Tooltip title={`${analysis.alerts_total ?? 0} total alerts`}>
            <Badge badgeContent={emergencyCount} color="error" max={99} sx={{ "& .MuiBadge-badge": { bgcolor: "#ff1744", boxShadow: "0 0 15px rgba(255,23,68,0.7)", fontSize: 13, fontWeight: 800 } }}>
              <Chip
                icon={<NotificationsActiveIcon sx={{ fontSize: 20 }} />}
                label={`${analysis.alerts_total ?? 0} Alerts`}
                sx={{ bgcolor: emergencyCount > 0 ? "rgba(255,23,68,0.2)" : "rgba(255,171,0,0.15)", color: emergencyCount > 0 ? "#ff5252" : "#ffab00", fontWeight: 800, fontSize: 14, height: 42, px: 1.5, border: `2px solid ${emergencyCount > 0 ? "rgba(255,23,68,0.5)" : "rgba(255,171,0,0.4)"}`, "& .MuiChip-icon": { color: emergencyCount > 0 ? "#ff5252" : "#ffab00" } }}
              />
            </Badge>
          </Tooltip>

          <Divider orientation="vertical" flexItem sx={{ mx: 3, borderColor: "rgba(0,217,255,0.3)", height: 55, alignSelf: "center" }} />

          {/* Actions */}
          <Tooltip title={soundEnabled ? "Mute alerts" : "Unmute alerts"}>
            <IconButton
              onClick={() => setSoundEnabled(s => !s)}
              sx={{ color: soundEnabled ? "#00ff88" : "rgba(255,255,255,0.4)", bgcolor: soundEnabled ? "rgba(0,255,136,0.15)" : "rgba(255,255,255,0.1)", border: `2px solid ${soundEnabled ? "rgba(0,255,136,0.4)" : "rgba(255,255,255,0.2)"}`, width: 48, height: 48, "&:hover": { color: soundEnabled ? "#00ff88" : "#00d9ff", bgcolor: soundEnabled ? "rgba(0,255,136,0.25)" : "rgba(0,217,255,0.15)", borderColor: soundEnabled ? "rgba(0,255,136,0.6)" : "rgba(0,217,255,0.5)" } }}
            >
              {soundEnabled ? <VolumeUpIcon sx={{ fontSize: 24 }} /> : <VolumeOffIcon sx={{ fontSize: 24 }} />}
            </IconButton>
          </Tooltip>

          <Tooltip title={isFullscreen ? "Exit fullscreen" : "Fullscreen mode"}>
            <IconButton
              onClick={toggleFullscreen}
              sx={{ color: "rgba(255,255,255,0.7)", bgcolor: "rgba(255,255,255,0.1)", border: "2px solid rgba(255,255,255,0.2)", width: 48, height: 48, "&:hover": { color: "#00d9ff", bgcolor: "rgba(0,217,255,0.15)", borderColor: "rgba(0,217,255,0.5)" } }}
            >
              {isFullscreen ? <FullscreenExitIcon sx={{ fontSize: 24 }} /> : <FullscreenIcon sx={{ fontSize: 24 }} />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Export alerts to CSV">
            <IconButton
              onClick={exportToCSV}
              sx={{ color: "rgba(255,255,255,0.7)", bgcolor: "rgba(255,255,255,0.1)", border: "2px solid rgba(255,255,255,0.2)", width: 48, height: 48, "&:hover": { color: "#00d9ff", bgcolor: "rgba(0,217,255,0.15)", borderColor: "rgba(0,217,255,0.5)" } }}
            >
              <DownloadIcon sx={{ fontSize: 24 }} />
            </IconButton>
          </Tooltip>

          <Tooltip title={autoAnnounce ? "Turn off auto announcements" : "Turn on auto announcements"}>
            <IconButton
              onClick={() => setAutoAnnounce(prev => !prev)}
              sx={{ 
                color: autoAnnounce ? (emergencyCount > 0 ? "#ff5252" : "#00ff88") : "rgba(255,255,255,0.4)", 
                bgcolor: autoAnnounce ? (emergencyCount > 0 ? "rgba(255,82,82,0.15)" : "rgba(0,255,136,0.15)") : "rgba(255,255,255,0.1)", 
                border: `2px solid ${autoAnnounce ? (emergencyCount > 0 ? "rgba(255,82,82,0.4)" : "rgba(0,255,136,0.4)") : "rgba(255,255,255,0.2)"}`, 
                width: 48, 
                height: 48, 
                animation: autoAnnounce && emergencyCount > 0 ? "pulse 1.5s infinite" : "none",
                "&:hover": { 
                  color: autoAnnounce ? (emergencyCount > 0 ? "#ff5252" : "#00ff88") : "#00d9ff", 
                  bgcolor: autoAnnounce ? (emergencyCount > 0 ? "rgba(255,82,82,0.25)" : "rgba(0,255,136,0.25)") : "rgba(0,217,255,0.15)", 
                  borderColor: autoAnnounce ? (emergencyCount > 0 ? "rgba(255,82,82,0.6)" : "rgba(0,255,136,0.6)") : "rgba(0,217,255,0.5)" 
                } 
              }}
            >
              <NotificationsActiveIcon sx={{ fontSize: 24 }} />
            </IconButton>
          </Tooltip>

          <Tooltip title="Reload zones">
            <IconButton
              onClick={() => fetch(`${API}/api/zones/reload`, { method: "POST" }).catch(() => {})}
              sx={{ color: "rgba(255,255,255,0.7)", bgcolor: "rgba(255,255,255,0.1)", border: "2px solid rgba(255,255,255,0.2)", width: 48, height: 48, "&:hover": { color: "#00d9ff", bgcolor: "rgba(0,217,255,0.15)", borderColor: "rgba(0,217,255,0.5)" } }}
            >
              <RefreshIcon sx={{ fontSize: 24 }} />
            </IconButton>
          </Tooltip>

          <Button
            variant={paused ? "outlined" : "contained"}
            startIcon={paused ? <PlayArrowIcon /> : <PauseIcon />}
            onClick={() => {
              setPaused((p) => {
                if (!p && 'speechSynthesis' in window) {
                  window.speechSynthesis.cancel(); // Stop any ongoing speech when pausing
                }
                return !p;
              });
            }}
            sx={{ background: paused ? "transparent" : "linear-gradient(135deg, #00d9ff 0%, #0096c7 100%)", borderColor: "#00d9ff", borderWidth: 2, color: paused ? "#00d9ff" : "#0d1b2a", fontWeight: 800, fontSize: 15, textTransform: "none", px: 4, py: 1.3, borderRadius: 2, boxShadow: paused ? "none" : "0 4px 30px rgba(0,217,255,0.5)", "&:hover": { borderWidth: 2, background: paused ? "rgba(0,217,255,0.15)" : "linear-gradient(135deg, #0096c7 0%, #0077b6 100%)" } }}
          >
            {paused ? "Resume" : "Pause All"}
          </Button>
        </Toolbar>

        {/* Tabs - Separated section with subtle glassmorphism */}
        <Box sx={{ bgcolor: "rgba(10,20,32,0.95)", borderRadius: { xs: 0, md: "0 0 14px 14px" }, position: "relative", "&::before": { content: '""', position: "absolute", top: 0, left: "3%", right: "3%", height: "1px", background: "linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 20%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0.1) 80%, transparent 100%)" } }}>
          <Tabs
            value={tab}
            onChange={(_, v) => setTab(v)}
            textColor="inherit"
            TabIndicatorProps={{ style: { background: "linear-gradient(90deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0.2) 100%)", height: 2, borderRadius: 2 } }}
            sx={{ px: { xs: 2, md: 5 }, minHeight: 58, "& .MuiTab-root": { minHeight: 58, textTransform: "none", fontWeight: 700, fontSize: 15, color: "rgba(255,255,255,0.5)", gap: 1.5, px: 4, transition: "all 0.2s", "&:hover": { color: "#fff", bgcolor: "rgba(255,255,255,0.03)" }, "&.Mui-selected": { color: "#fff" } } }}
          >
            <Tab icon={<DashboardIcon sx={{ fontSize: 22 }} />} iconPosition="start" label="Dashboard" />
            <Tab icon={<AnalyticsIcon sx={{ fontSize: 22 }} />} iconPosition="start" label="Analytics" />
            <Tab icon={<HistoryIcon sx={{ fontSize: 22 }} />} iconPosition="start" label="Event Logs" />
            <Tab icon={<SettingsIcon sx={{ fontSize: 22 }} />} iconPosition="start" label="Settings" />
          </Tabs>
        </Box>
      </AppBar>

      {/* CONTENT AREA - Lighter background */}
      <Box sx={{ p: { xs: 2, md: 5 }, pt: { xs: 3, md: 5 }, maxWidth: 2000, mx: "auto", minHeight: "calc(100vh - 180px)" }}>
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
                  sx={{ p: 3.5, borderRadius: 3, bgcolor: "#161d28", border: "1px solid rgba(255,255,255,0.1)", display: "flex", alignItems: "center", gap: 3, transition: "all 0.3s", "&:hover": { bgcolor: "#1a242f", borderColor: `${stat.color}60`, transform: "translateY(-3px)", boxShadow: `0 12px 40px ${stat.color}20` } }}
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
                  <VideocamIcon sx={{ color: "#00d9ff", fontSize: 32 }} />
                  Live Zone Feeds
                </Typography>
                <Typography sx={{ fontSize: 15, color: "rgba(255,255,255,0.5)", mt: 0.5, ml: 6 }}>Real-time surveillance monitoring across all beach zones</Typography>
              </Box>
              <Chip label={`${zones.length} Active`} sx={{ bgcolor: "rgba(0,217,255,0.12)", color: "#00d9ff", fontWeight: 800, fontSize: 14, height: 36, px: 1, border: "2px solid rgba(0,217,255,0.4)" }} />
            </Box>

            {/* Zone Grid */}
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", lg: "repeat(3, 1fr)", xl: "repeat(3, 1fr)" }, gap: 4 }}>
            {zones.length === 0 ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 3, alignItems: "center", justifyContent: "center", textAlign: "center", gridColumn: "1/-1", py: 14, borderRadius: 4, bgcolor: "#141e29", border: "2px dashed rgba(0,217,255,0.35)" }}>
                <Avatar sx={{ bgcolor: "rgba(0,217,255,0.15)", width: 90, height: 90, boxShadow: "0 0 40px rgba(0,217,255,0.3)" }}>
                  <VideocamIcon sx={{ fontSize: 45, color: "#00d9ff" }} />
                </Avatar>
                <Box>
                  <Typography sx={{ fontWeight: 900, fontSize: 26, color: "#fff" }}>No Zones Detected</Typography>
                  <Typography sx={{ color: "rgba(255,255,255,0.5)", maxWidth: 480, mt: 1.5, fontSize: 16 }}>Add video files named zone1.mp4, zone2.mp4, etc. to the videos folder, then click the refresh button.</Typography>
                </Box>
                <Button startIcon={<RefreshIcon />} variant="contained" onClick={() => fetch(`${API}/api/zones/reload`, { method: "POST" })} sx={{ mt: 2, background: "linear-gradient(135deg, #00d9ff 0%, #0096c7 100%)", color: "#0d1b2a", fontWeight: 800, px: 5, py: 1.5, fontSize: 15, boxShadow: "0 4px 30px rgba(0,217,255,0.4)", "&:hover": { background: "linear-gradient(135deg, #0096c7 0%, #0077b6 100%)" } }}>Reload Zones</Button>
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
                    onClick={() => setOpenZone(z)}
                    sx={{
                      cursor: "pointer",
                      bgcolor: "#141e29",
                      border: "1px solid rgba(0,217,255,0.15)",
                      borderRadius: 4,
                      overflow: "hidden",
                      transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                      "&:hover": { borderColor: "rgba(0,217,255,0.6)", transform: "translateY(-5px)", boxShadow: "0 25px 70px rgba(0,217,255,0.2)" },
                    }}
                  >
                    {/* Zone Header with Label */}
                    <Box sx={{ p: 2.5, borderBottom: "1px solid rgba(0,217,255,0.1)", display: "flex", alignItems: "center", justifyContent: "space-between", bgcolor: "rgba(0,217,255,0.03)" }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                        <Avatar sx={{ background: "linear-gradient(135deg, #00d9ff 0%, #0096c7 100%)", width: 48, height: 48, fontSize: 18, fontWeight: 900, boxShadow: "0 4px 20px rgba(0,217,255,0.4)" }}>{z}</Avatar>
                        <Box>
                          <Typography sx={{ fontWeight: 900, fontSize: 20, letterSpacing: -0.3, color: "#fff" }}>Zone {z}</Typography>
                          <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.5)", fontWeight: 600 }}>{exists ? "Live Surveillance Feed" : "No Video Source"}</Typography>
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
                        <Box sx={{ position: "absolute", inset: 0, zIndex: 4, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", bgcolor: "rgba(6,13,24,.95)", px: 3, textAlign: "center" }}>
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
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 4 }}>
              <Box>
                <Typography sx={{ fontWeight: 900, fontSize: 28, color: "#fff", display: "flex", alignItems: "center", gap: 2 }}>
                  <Box sx={{ p: 1.5, borderRadius: 2, background: "linear-gradient(135deg, rgba(0,217,255,0.2) 0%, rgba(0,217,255,0.05) 100%)", display: "flex", border: "1px solid rgba(0,217,255,0.2)" }}>
                    <AnalyticsIcon sx={{ color: "#00d9ff", fontSize: 28 }} />
                  </Box>
                  Analytics Dashboard
                </Typography>
                <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 14, mt: 1, ml: 7 }}>Real-time surveillance insights</Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 2, py: 1, borderRadius: 2, bgcolor: "rgba(0,255,136,0.08)", border: "1px solid rgba(0,255,136,0.2)" }}>
                <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: "#00ff88", animation: "pulse 1.5s infinite" }} />
                <Typography sx={{ fontSize: 12, color: "#00ff88", fontWeight: 600 }}>LIVE</Typography>
              </Box>
            </Box>

            {/* Key Metrics Row */}
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "repeat(2, 1fr)", lg: "repeat(4, 1fr)" }, gap: 3, mb: 4 }}>
              {[
                { title: "Total Detections", value: analysis.alerts_total ?? 0, icon: "📊", gradient: "linear-gradient(135deg, #00d9ff 0%, #0099cc 100%)" },
                { title: "Monitored Zones", value: zones.length, icon: "🎯", gradient: "linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)" },
                { title: "Emergency Alerts", value: emergencyCount, icon: "🚨", gradient: "linear-gradient(135deg, #ff5252 0%, #cc4141 100%)" },
                { title: "Avg Confidence", value: `${((alerts.items?.reduce((a, b) => a + (b.conf || 0), 0) / Math.max(1, alerts.items?.length || 1)) * 100).toFixed(0)}%`, icon: "⚡", gradient: "linear-gradient(135deg, #ffab00 0%, #ff8f00 100%)" },
              ].map((card) => (
                <Box key={card.title} sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", transition: "all 0.3s ease", "&:hover": { transform: "translateY(-4px)", boxShadow: "0 12px 40px rgba(0,0,0,0.4)" } }}>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
                    <Box sx={{ p: 1.5, borderRadius: 2, background: card.gradient, boxShadow: "0 4px 15px rgba(0,0,0,0.3)" }}>
                      <Typography sx={{ fontSize: 20 }}>{card.icon}</Typography>
                    </Box>
                  </Box>
                  <Typography sx={{ fontSize: 36, fontWeight: 900, color: "#fff", lineHeight: 1 }}>{card.value}</Typography>
                  <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.5)", fontWeight: 600, mt: 1 }}>{card.title}</Typography>
                </Box>
              ))}
            </Box>

            {/* Main Charts Grid - Pie Chart and Bar Chart */}
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "400px 1fr" }, gap: 4, mb: 4 }}>
              
              {/* Professional Pie Chart - Detection Types */}
              <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)" }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 4 }}>
                  <Typography sx={{ fontWeight: 800, fontSize: 18, color: "#fff" }}>Detection Types</Typography>
                  <Chip label="Distribution" size="small" sx={{ bgcolor: "rgba(0,217,255,0.1)", color: "#00d9ff", fontWeight: 600, fontSize: 10, height: 24 }} />
                </Box>
                
                {Object.keys(analysis.alerts_by_label || {}).length > 0 ? (
                  <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
                    {/* SVG Pie Chart */}
                    <Box sx={{ position: "relative", width: 180, height: 180 }}>
                      <svg width="180" height="180" viewBox="0 0 100 100">
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
                          const colors = ["#00d9ff", "#ff5252", "#00ff88", "#ffab00", "#a855f7"];
                          let cumulative = 0;
                          return entries.map(([label, count], i) => {
                            const pct = (count / total) * 100;
                            const offset = cumulative;
                            cumulative += pct;
                            return (
                              <circle
                                key={label}
                                cx="50"
                                cy="50"
                                r="40"
                                fill="transparent"
                                stroke={colors[i % colors.length]}
                                strokeWidth="16"
                                strokeDasharray={`${pct * 2.51} ${251 - pct * 2.51}`}
                                strokeDashoffset={-offset * 2.51 + 62.75}
                                filter="url(#pieGlow)"
                                style={{ transition: "all 0.5s ease" }}
                              />
                            );
                          });
                        })()}
                        <circle cx="50" cy="50" r="30" fill="#0f1923" />
                      </svg>
                      <Box sx={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                        <Typography sx={{ fontSize: 32, fontWeight: 900, color: "#fff", lineHeight: 1 }}>{analysis.alerts_total ?? 0}</Typography>
                        <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.4)", fontWeight: 600, mt: 0.5 }}>TOTAL</Typography>
                      </Box>
                    </Box>
                    
                    {/* Legend */}
                    <Stack spacing={1} sx={{ width: "100%" }}>
                      {(() => {
                        const entries = Object.entries(analysis.alerts_by_label || {});
                        const total = entries.reduce((a, [, v]) => a + v, 0);
                        const colors = ["#00d9ff", "#ff5252", "#00ff88", "#ffab00", "#a855f7"];
                        return entries.map(([label, count], i) => {
                          const pct = ((count / total) * 100).toFixed(0);
                          const isEmergency = label.toLowerCase().includes("drown") || label.toLowerCase().includes("emerg");
                          return (
                            <Box key={label} sx={{ display: "flex", alignItems: "center", gap: 2, p: 1.5, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)", transition: "all 0.2s", "&:hover": { bgcolor: "rgba(255,255,255,0.04)" } }}>
                              <Box sx={{ width: 10, height: 10, borderRadius: 1, bgcolor: colors[i % colors.length] }} />
                              <Typography sx={{ flex: 1, fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.8)", textTransform: "capitalize" }}>{label}</Typography>
                              <Typography sx={{ fontSize: 14, fontWeight: 800, color: colors[i % colors.length] }}>{count}</Typography>
                              <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.4)", minWidth: 35 }}>{pct}%</Typography>
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

              {/* Professional Bar Chart - Zone Activity */}
              <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)" }}>
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 4 }}>
                  <Typography sx={{ fontWeight: 800, fontSize: 18, color: "#fff" }}>Zone Activity</Typography>
                  <Chip label={`${zones.length} Zones`} size="small" sx={{ bgcolor: "rgba(0,255,136,0.1)", color: "#00ff88", fontWeight: 600, fontSize: 10, height: 24 }} />
                </Box>
                
                {zones.length > 0 ? (
                  <Box>
                    <Box sx={{ display: "flex", gap: 2, height: 260 }}>
                      <Box sx={{ display: "flex", flexDirection: "column", justifyContent: "space-between", py: 1, pr: 1 }}>
                        {(() => {
                          const maxCount = Math.max(...Object.values(analysis.alerts_by_zone || { default: 0 }), 1);
                          return [maxCount, Math.round(maxCount * 0.75), Math.round(maxCount * 0.5), Math.round(maxCount * 0.25), 0].map((val) => (
                            <Typography key={val} sx={{ fontSize: 10, color: "rgba(255,255,255,0.3)", minWidth: 20, textAlign: "right" }}>{val}</Typography>
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
                            const colors = ["#00d9ff", "#00ff88", "#ffab00", "#a855f7", "#f472b6", "#22d3ee"];
                            return zones.map((zone, idx) => {
                              const count = (analysis.alerts_by_zone || {})[zone] || 0;
                              const height = (count / maxCount) * 100;
                              const color = colors[idx % colors.length];
                              return (
                                <Box key={zone} sx={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" }}>
                                  <Box sx={{ flex: 1, width: "100%", display: "flex", alignItems: "flex-end", justifyContent: "center" }}>
                                    <Box sx={{ width: "70%", maxWidth: 50, height: `${height}%`, minHeight: count > 0 ? 8 : 2, background: `linear-gradient(180deg, ${color} 0%, ${color}70 100%)`, borderRadius: "4px 4px 0 0", boxShadow: count > 0 ? `0 0 20px ${color}30` : "none", transition: "all 0.5s ease", position: "relative", "&::before": count > 0 ? { content: '""', position: "absolute", top: 0, left: 0, right: 0, height: "40%", background: "linear-gradient(180deg, rgba(255,255,255,0.25) 0%, transparent 100%)", borderRadius: "4px 4px 0 0" } : {}, "&:hover": { transform: "scaleY(1.02)", boxShadow: `0 0 30px ${color}50` } }} />
                                  </Box>
                                </Box>
                              );
                            });
                          })()}
                        </Box>
                      </Box>
                    </Box>
                    <Box sx={{ display: "flex", pl: 4, mt: 1 }}>
                      {zones.map((zone, idx) => {
                        const count = (analysis.alerts_by_zone || {})[zone] || 0;
                        const colors = ["#00d9ff", "#00ff88", "#ffab00", "#a855f7", "#f472b6", "#22d3ee"];
                        return (
                          <Box key={zone} sx={{ flex: 1, textAlign: "center" }}>
                            <Typography sx={{ fontSize: 11, color: colors[idx % colors.length], fontWeight: 700 }}>Z{zone}</Typography>
                            <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.4)", mt: 0.25 }}>{count}</Typography>
                          </Box>
                        );
                      })}
                    </Box>
                    <Box sx={{ display: "flex", gap: 2, mt: 3, pt: 3, borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                      {[
                        { label: "Most Active", value: Object.entries(analysis.alerts_by_zone || {}).sort((a, b) => b[1] - a[1])[0]?.[0] ? `Zone ${Object.entries(analysis.alerts_by_zone || {}).sort((a, b) => b[1] - a[1])[0]?.[0]}` : "—", color: "#00d9ff" },
                        { label: "Avg/Zone", value: zones.length > 0 ? ((analysis.alerts_total || 0) / zones.length).toFixed(1) : "0", color: "#00ff88" },
                        { label: "Coverage", value: zones.length > 0 ? `${((Object.keys(analysis.alerts_by_zone || {}).length / zones.length) * 100).toFixed(0)}%` : "0%", color: "#ffab00" },
                      ].map((stat) => (
                        <Box key={stat.label} sx={{ flex: 1, p: 2, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", textAlign: "center" }}>
                          <Typography sx={{ fontSize: 16, fontWeight: 800, color: stat.color }}>{stat.value}</Typography>
                          <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.4)", mt: 0.5 }}>{stat.label}</Typography>
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

            {/* Live Activity Feed */}
            <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)" }}>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <Typography sx={{ fontWeight: 800, fontSize: 18, color: "#fff" }}>Recent Activity</Typography>
                  <Chip label={`${(alerts.items || []).length} Events`} size="small" sx={{ bgcolor: "rgba(255,255,255,0.05)", color: "rgba(255,255,255,0.6)", fontWeight: 600, fontSize: 10, height: 22 }} />
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "#00ff88", animation: "pulse 1.5s infinite" }} />
                  <Typography sx={{ fontSize: 11, color: "#00ff88", fontWeight: 600 }}>LIVE</Typography>
                </Box>
              </Box>
              
              <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)", xl: "repeat(3, 1fr)" }, gap: 2, maxHeight: 280, overflowY: "auto", pr: 1, "&::-webkit-scrollbar": { width: 4 }, "&::-webkit-scrollbar-thumb": { bgcolor: "rgba(255,255,255,0.1)", borderRadius: 2 } }}>
                {(alerts.items || []).length > 0 ? (
                  (alerts.items || []).slice(0, 12).map((alert, idx) => {
                    const isEmergency = String(alert.label || "").toLowerCase().includes("drown") || String(alert.label || "").toLowerCase().includes("emerg");
                    const color = isEmergency ? "#ff5252" : "#00d9ff";
                    return (
                      <Box key={idx} sx={{ display: "flex", alignItems: "center", gap: 2, p: 2, borderRadius: 2, bgcolor: isEmergency ? "rgba(255,82,82,0.06)" : "rgba(255,255,255,0.02)", border: `1px solid ${isEmergency ? "rgba(255,82,82,0.15)" : "rgba(255,255,255,0.04)"}`, transition: "all 0.2s", "&:hover": { bgcolor: isEmergency ? "rgba(255,82,82,0.1)" : "rgba(255,255,255,0.04)" } }}>
                        <Box sx={{ width: 3, height: 36, borderRadius: 1, bgcolor: color, flexShrink: 0 }} />
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                            <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#fff", textTransform: "capitalize" }}>{alert.label || "Detection"}</Typography>
                            <Chip label={`Z${alert.zone}`} size="small" sx={{ height: 16, fontSize: 9, fontWeight: 700, bgcolor: "rgba(0,217,255,0.1)", color: "#00d9ff" }} />
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
          </Box>
        )}

        {tab === 2 && (
          <Box>
            <Typography sx={{ fontWeight: 800, fontSize: 20, mb: 3 }}>Event History</Typography>
            <Box sx={{ borderRadius: 2, bgcolor: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", overflow: "hidden" }}>
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
              <Box sx={{ p: 3, borderRadius: 2, bgcolor: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <Typography sx={{ fontWeight: 700, mb: 2, color: "#00bcd4" }}>Backend Configuration</Typography>
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

              <Box sx={{ p: 3, borderRadius: 2, bgcolor: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <Typography sx={{ fontWeight: 700, mb: 2, color: "#ff9800" }}>Detection Settings</Typography>
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
              <Box sx={{ p: 3, borderRadius: 2, bgcolor: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", gridColumn: { md: "span 2" } }}>
                <Typography sx={{ fontWeight: 700, mb: 2, color: "#00ff88" }}>Voice Alert Settings</Typography>
                <Stack spacing={2}>
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", py: 1, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                    <Box>
                      <Typography sx={{ fontSize: 14, fontWeight: 600 }}>Emergency Voice Alerts</Typography>
                      <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.5)" }}>Play alarm and voice announcement when drowning is detected</Typography>
                    </Box>
                    <Button
                      variant={soundEnabled ? "contained" : "outlined"}
                      onClick={() => setSoundEnabled(s => !s)}
                      startIcon={soundEnabled ? <VolumeUpIcon /> : <VolumeOffIcon />}
                      sx={{ 
                        background: soundEnabled ? "linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)" : "transparent",
                        borderColor: "#00ff88",
                        color: soundEnabled ? "#0a1117" : "#00ff88",
                        fontWeight: 700,
                        textTransform: "none",
                        "&:hover": { background: soundEnabled ? "linear-gradient(135deg, #00cc6a 0%, #00aa55 100%)" : "rgba(0,255,136,0.15)" }
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
                        // Speak test message
                        if ('speechSynthesis' in window) {
                          window.speechSynthesis.cancel();
                          const utterance = new SpeechSynthesisUtterance("Alert! Drowning detected in Zone 1. Please check immediately.");
                          utterance.rate = 1.0;
                          utterance.volume = 1.0;
                          utterance.lang = 'en-US';
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
            </Box>
          </Box>
        )}
      </Box>

      <Dialog
        open={!!openZone}
        onClose={() => setOpenZone(null)}
        maxWidth="xl"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: "#0a1117",
            borderRadius: 4,
            border: "1px solid rgba(0,217,255,0.2)",
            p: 0,
            overflow: "hidden",
            width: { xs: "100vw", md: "95vw" },
            height: { xs: "100vh", md: "92vh" },
            maxWidth: "none",
            m: { xs: 0, md: 2 },
            boxShadow: "0 25px 80px rgba(0,0,0,0.6), 0 0 100px rgba(0,217,255,0.1)",
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
            background: "linear-gradient(180deg, rgba(0,217,255,0.08) 0%, transparent 100%)",
            borderBottom: "1px solid rgba(0,217,255,0.15)"
          }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
              <Box sx={{ position: "relative" }}>
                <Avatar sx={{ 
                  background: "linear-gradient(135deg, #00d9ff 0%, #0096c7 100%)", 
                  fontWeight: 900, 
                  width: 56, 
                  height: 56,
                  fontSize: 24,
                  boxShadow: "0 0 30px rgba(0,217,255,0.5)",
                  border: "3px solid rgba(0,217,255,0.3)"
                }}>{openZone}</Avatar>
                <Box sx={{ 
                  position: "absolute", 
                  bottom: 2, 
                  right: 2, 
                  width: 14, 
                  height: 14, 
                  borderRadius: "50%", 
                  bgcolor: modalPaused ? "#ffab00" : "#00ff88", 
                  border: "2px solid #0a1117",
                  boxShadow: `0 0 10px ${modalPaused ? "rgba(255,171,0,0.6)" : "rgba(0,255,136,0.6)"}`
                }} />
              </Box>
              <Box>
                <Typography sx={{ fontWeight: 900, fontSize: 26, background: "linear-gradient(135deg, #fff 0%, #00d9ff 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Zone {openZone}</Typography>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 0.5 }}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <MyLocationIcon sx={{ fontSize: 14, color: "#00d9ff" }} />
                    <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.6)" }}>Live Monitoring</Typography>
                  </Box>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                    <SpeedIcon sx={{ fontSize: 14, color: "#00ff88" }} />
                    <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.6)" }}>Real-time AI</Typography>
                  </Box>
                </Stack>
              </Box>
            </Box>
            
            <Stack direction="row" spacing={1.5} alignItems="center">
              <Chip 
                icon={<AccessTimeIcon sx={{ fontSize: 16 }} />}
                label={modalDetections.age_s != null ? `${modalDetections.age_s.toFixed(1)}s ago` : "Live"}
                sx={{ bgcolor: "rgba(0,255,136,0.12)", color: "#00ff88", fontWeight: 700, fontSize: 12, border: "1px solid rgba(0,255,136,0.3)", "& .MuiChip-icon": { color: "#00ff88" } }}
              />
              <Chip 
                icon={<CenterFocusStrongIcon sx={{ fontSize: 16 }} />}
                label={`${modalDetections.count ?? 0} Detected`}
                sx={{ bgcolor: "rgba(0,217,255,0.12)", color: "#00d9ff", fontWeight: 700, fontSize: 12, border: "1px solid rgba(0,217,255,0.3)", "& .MuiChip-icon": { color: "#00d9ff" } }}
              />
              <Box sx={{ width: 1, height: 32, bgcolor: "rgba(255,255,255,0.1)", mx: 1 }} />
              
              {/* Sound Toggle */}
              <Tooltip title={soundEnabled ? "Mute voice alerts" : "Enable voice alerts"}>
                <IconButton 
                  onClick={() => setSoundEnabled(s => !s)}
                  sx={{ 
                    color: soundEnabled ? "#00ff88" : "rgba(255,255,255,0.4)", 
                    bgcolor: soundEnabled ? "rgba(0,255,136,0.15)" : "rgba(255,255,255,0.1)",
                    border: `1px solid ${soundEnabled ? "rgba(0,255,136,0.3)" : "rgba(255,255,255,0.2)"}`,
                    "&:hover": { bgcolor: soundEnabled ? "rgba(0,255,136,0.25)" : "rgba(255,255,255,0.15)" }
                  }}
                >
                  {soundEnabled ? <VolumeUpIcon /> : <VolumeOffIcon />}
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Zoom hint: Scroll to zoom, drag to pan">
                <IconButton sx={{ color: "rgba(255,255,255,0.5)", "&:hover": { color: "#00d9ff" } }}>
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
                  background: modalPaused ? "transparent" : "linear-gradient(135deg, #00d9ff 0%, #0096c7 100%)", 
                  borderColor: "#00d9ff", 
                  borderWidth: 2,
                  color: modalPaused ? "#00d9ff" : "#0a1117", 
                  fontWeight: 800, 
                  textTransform: "none",
                  px: 4,
                  py: 1.2,
                  fontSize: 16,
                  minWidth: 140,
                  boxShadow: modalPaused ? "none" : "0 4px 20px rgba(0,217,255,0.4)",
                  "&:hover": { borderWidth: 2, background: modalPaused ? "rgba(0,217,255,0.15)" : "linear-gradient(135deg, #0096c7 0%, #0077b6 100%)" }
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
                  background: "linear-gradient(135deg, #0d1b2a 0%, #000 100%)",
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
                        <img
                          src={openZone && modalUseMjpeg && !modalPaused ? `${API}/api/zones/${openZone}/stream.mjpg` : ""}
                          alt=""
                          onLoad={() => setModalMjpegOk(true)}
                          onError={() => { setModalUseMjpeg(false); setModalMjpegOk(false); }}
                          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", display: openZone && modalUseMjpeg && !modalPaused ? "block" : "none" }}
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
                            sx={{ bgcolor: "rgba(0,0,0,0.7)", color: "#00d9ff", "&:hover": { bgcolor: "rgba(0,217,255,0.2)" } }}
                          >
                            <ZoomInIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Zoom Out" placement="left">
                          <IconButton 
                            onClick={() => zoomOut()} 
                            sx={{ bgcolor: "rgba(0,0,0,0.7)", color: "#00d9ff", "&:hover": { bgcolor: "rgba(0,217,255,0.2)" } }}
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
                  border: "1px solid rgba(0,217,255,0.3)",
                  backdropFilter: "blur(10px)",
                  display: "flex",
                  alignItems: "center",
                  gap: 1
                }}>
                  <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: modalPaused ? "#ffab00" : "#00ff88", animation: modalPaused ? "none" : "pulse 2s infinite" }} />
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
                  border: "1px solid rgba(0,217,255,0.3)",
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
                      bgcolor: modalPaused ? "rgba(0,217,255,0.2)" : "linear-gradient(135deg, #00d9ff 0%, #0096c7 100%)",
                      background: modalPaused ? "rgba(0,217,255,0.2)" : "linear-gradient(135deg, #00d9ff 0%, #0096c7 100%)",
                      color: modalPaused ? "#00d9ff" : "#0a1117",
                      border: "2px solid rgba(0,217,255,0.5)",
                      boxShadow: modalPaused ? "none" : "0 0 30px rgba(0,217,255,0.5)",
                      "&:hover": { 
                        bgcolor: modalPaused ? "rgba(0,217,255,0.35)" : "#0096c7",
                        background: modalPaused ? "rgba(0,217,255,0.35)" : "#0096c7",
                      }
                    }}
                  >
                    {modalPaused ? <PlayArrowIcon sx={{ fontSize: 36 }} /> : <PauseIcon sx={{ fontSize: 36 }} />}
                  </IconButton>
                  
                  <Box sx={{ width: 1, height: 40, bgcolor: "rgba(255,255,255,0.15)" }} />
                  
                  {/* Detection count */}
                  <Box sx={{ textAlign: "center", minWidth: 80 }}>
                    <Typography sx={{ fontSize: 24, fontWeight: 900, color: "#00d9ff", lineHeight: 1 }}>{modalDetections.count ?? 0}</Typography>
                    <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.5)", textTransform: "uppercase", letterSpacing: 1 }}>Detected</Typography>
                  </Box>
                  
                  <Box sx={{ width: 1, height: 40, bgcolor: "rgba(255,255,255,0.15)" }} />
                  
                  {/* Sound toggle */}
                  <Tooltip title={soundEnabled ? "Mute" : "Unmute"}>
                    <IconButton 
                      onClick={() => setSoundEnabled(s => !s)}
                      sx={{ 
                        color: soundEnabled ? "#00ff88" : "rgba(255,255,255,0.4)", 
                        "&:hover": { color: soundEnabled ? "#00ff88" : "#fff" }
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
            <Box sx={{ minHeight: 0, display: "flex", flexDirection: "column", gap: 2, overflow: "hidden", display: { xs: "none", lg: "flex" } }}>
              {/* Live Stats Card */}
              <Box sx={{ 
                p: 2.5, 
                borderRadius: 3, 
                background: "linear-gradient(135deg, rgba(0,217,255,0.15) 0%, rgba(0,150,199,0.08) 100%)",
                border: "1px solid rgba(0,217,255,0.25)",
                position: "relative",
                overflow: "hidden"
              }}>
                <Box sx={{ position: "absolute", top: -30, right: -30, width: 100, height: 100, borderRadius: "50%", bgcolor: "rgba(0,217,255,0.1)" }} />
                <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 800, textTransform: "uppercase", letterSpacing: 1.5, mb: 1 }}>Live Detection</Typography>
                <Box sx={{ display: "flex", alignItems: "baseline", gap: 1 }}>
                  <Typography sx={{ fontSize: 48, fontWeight: 900, color: "#00d9ff", lineHeight: 1 }}>{modalDetections.count ?? 0}</Typography>
                  <Typography sx={{ fontSize: 16, color: "rgba(255,255,255,0.5)", fontWeight: 600 }}>objects</Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1.5 }}>
                  <TrendingUpIcon sx={{ fontSize: 16, color: "#00ff88" }} />
                  <Typography sx={{ fontSize: 12, color: "#00ff88", fontWeight: 600 }}>Active monitoring</Typography>
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
                    { label: "Detection Objects", value: Object.keys(modalAnalysis.alerts_by_label || {}).length, color: "#00d9ff" },
                    { label: "Stream Status", value: modalMjpegOk ? "Active" : "Polling", color: "#00ff88" },
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
                <Box sx={{ flex: 1, overflow: "auto", pr: 1, "&::-webkit-scrollbar": { width: 4 }, "&::-webkit-scrollbar-track": { bgcolor: "rgba(255,255,255,0.05)", borderRadius: 2 }, "&::-webkit-scrollbar-thumb": { bgcolor: "rgba(0,217,255,0.3)", borderRadius: 2 } }}>
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
                                bgcolor: isEmergency ? "rgba(255,82,82,0.2)" : "rgba(0,217,255,0.15)", 
                                color: isEmergency ? "#ff5252" : "#00d9ff", 
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
