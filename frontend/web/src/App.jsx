import React, { useEffect, useMemo, useRef, useState } from "react";
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
  CardMedia,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const MODAL_REFRESH_MS = 250; // slower polling for fallback frame mode (reduces aborted-request false errors)

function usePollJson(url, ms, enabled, initial) {
  const [data, setData] = useState(initial);
  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(url);
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

export default function App() {
  const zones = useMemo(() => Array.from({ length: 6 }, (_, i) => i + 1), []);
  const [tab, setTab] = useState(0);
  const [paused, setPaused] = useState(false);
  const [tick, setTick] = useState(0);
  const [openZone, setOpenZone] = useState(null);
  const [imgErr, setImgErr] = useState({});
  const [modalPaused, setModalPaused] = useState(false);
  const [streamDisabled, setStreamDisabled] = useState({}); // { [zoneId]: true } => fallback to frame.jpg
  const [modalBackendOk, setModalBackendOk] = useState(null); // null | true | false
  const [modalBlobUrl, setModalBlobUrl] = useState("");
  const modalBlobUrlRef = useRef("");
  const [modalUseMjpeg, setModalUseMjpeg] = useState(true);
  const [modalMjpegOk, setModalMjpegOk] = useState(false);
  const modalVideoBoxRef = useRef(null);

  // Drive cache-busting ticks when paused OR when we fallback to frame polling
  useEffect(() => {
    const needTick = paused || Object.values(streamDisabled).some(Boolean);
    if (!needTick) return;
    const t = setInterval(() => setTick((x) => x + 1), 180);
    return () => clearInterval(t);
  }, [paused, streamDisabled]);

  useEffect(() => {
    if (!openZone) {
      setModalPaused(false);
      setModalUseMjpeg(true);
      setModalMjpegOk(false);
      if (modalBlobUrlRef.current) {
        URL.revokeObjectURL(modalBlobUrlRef.current);
        modalBlobUrlRef.current = "";
      }
      setModalBlobUrl("");
    } else {
      setModalUseMjpeg(true);
      setModalMjpegOk(false);
      if (modalBlobUrlRef.current) {
        URL.revokeObjectURL(modalBlobUrlRef.current);
        modalBlobUrlRef.current = "";
      }
      setModalBlobUrl("");
    }
  }, [openZone]);

  // If MJPEG doesn't start quickly in the modal, force reliable fallback.
  useEffect(() => {
    if (!openZone) return;
    if (modalPaused) return;
    if (!modalUseMjpeg) return;
    if (modalMjpegOk) return;

    const t = setTimeout(() => {
      // No first-frame from MJPEG -> stay on fallback.
      setModalUseMjpeg(false);
    }, 2000);
    return () => clearTimeout(t);
  }, [openZone, modalPaused, modalUseMjpeg, modalMjpegOk]);

  // Prevent the dialog/page from scrolling when interacting with the zoom area.
  // Drag should pan the video; wheel/pinch should zoom the video.
  useEffect(() => {
    const el = modalVideoBoxRef.current;
    if (!openZone || !el) return;

    const onWheel = (e) => {
      e.preventDefault();
    };
    const onTouchMove = (e) => {
      e.preventDefault();
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("touchmove", onTouchMove, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("touchmove", onTouchMove);
    };
  }, [openZone]);

  const zoneSrc = (z) => {
    const fallback = paused || !!streamDisabled[z];
    return fallback ? `${API}/api/zones/${z}/frame.jpg?t=${tick}` : `${API}/api/zones/${z}/stream.mjpg`;
  };

  // Modal fallback playback:
  // Always keep a blob-based frame available so the modal never looks blank.
  // When MJPEG is healthy it will visually take over (and we stop polling blobs to save bandwidth).
  useEffect(() => {
    if (!openZone) return;

    let alive = true;
    let inFlight = false;
    const ctrl = new AbortController();

    const shouldPoll = modalPaused || !modalUseMjpeg || !modalMjpegOk;
    if (!shouldPoll) {
      return () => {
        alive = false;
        ctrl.abort();
      };
    }

    const fetchFrame = async () => {
      if (!alive) return;
      // When paused, fetch at most once (freeze frame)
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
        // ignore transient fetch aborts/errors; keep the last good frame
      } finally {
        inFlight = false;
      }
    };

    fetchFrame();
    const t = modalPaused ? null : setInterval(fetchFrame, MODAL_REFRESH_MS);

    return () => {
      alive = false;
      if (t) clearInterval(t);
      ctrl.abort();
    };
  }, [openZone, modalPaused, modalUseMjpeg, modalMjpegOk]);

  const alerts = usePollJson(`${API}/api/alerts?limit=120`, 1000, true, { items: [] });
  const analysis = usePollJson(`${API}/api/analysis`, 1500, true, {
    alerts_total: 0,
    alerts_by_zone: {},
    alerts_by_label: {},
  });

  const modalAlerts = usePollJson(
    openZone ? `${API}/api/alerts?zone=${openZone}&limit=40` : `${API}/api/alerts?limit=1`,
    900,
    !!openZone,
    { items: [] }
  );
  const modalAnalysis = usePollJson(
    openZone ? `${API}/api/analysis?zone=${openZone}` : `${API}/api/analysis`,
    1200,
    !!openZone,
    { alerts_total: 0, alerts_by_zone: {}, alerts_by_label: {} }
  );

  const modalDetections = usePollJson(
    openZone ? `${API}/api/zones/${openZone}/detections` : `${API}/api/zones/1/detections`,
    250,
    !!openZone,
    { zone: null, count: 0, age_s: null, items: [] }
  );

  // NEW: verify backend can serve at least a single JPEG for this zone.
  // This avoids relying on <img onLoad> (which is unreliable with MJPEG and can be noisy when URLs change fast).
  useEffect(() => {
    if (!openZone) {
      setModalBackendOk(null);
      return;
    }

    let alive = true;
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 2000);

    fetch(`${API}/api/zones/${openZone}/frame.jpg?t=${Date.now()}`, { cache: "no-store", signal: ctrl.signal })
      .then((r) => alive && setModalBackendOk(r.ok))
      .catch(() => alive && setModalBackendOk(false))
      .finally(() => clearTimeout(timeout));

    return () => {
      alive = false;
      clearTimeout(timeout);
      ctrl.abort();
    };
  }, [openZone]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        bgcolor: "linear-gradient(135deg, #0b1220 60%, #1a233a 100%)",
        color: "#e7eefc",
        backgroundAttachment: "fixed",
      }}
    >
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          bgcolor: "rgba(15,27,51,.85)",
          backdropFilter: "blur(16px)",
          boxShadow: "0 4px 32px 0 rgba(46,230,255,0.08)",
          borderBottom: "1.5px solid rgba(46,230,255,0.10)",
        }}
      >
        <Toolbar sx={{ gap: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: 0.5 }}>
            CoastVision
          </Typography>
          <Chip
            label={`Alerts: ${analysis.alerts_total ?? 0}`}
            size="small"
            sx={{ bgcolor: "rgba(46,230,255,.12)", color: "#2ee6ff" }}
          />
          <Box sx={{ flex: 1 }} />
          <Button
            variant={paused ? "outlined" : "contained"}
            startIcon={paused ? <PlayArrowIcon /> : <PauseIcon />}
            onClick={() => setPaused((p) => !p)}
            sx={{
              borderColor: "rgba(46,230,255,.35)",
              bgcolor: paused ? "transparent" : "#2ee6ff",
              color: paused ? "#2ee6ff" : "#0b1220",
              fontWeight: 800,
            }}
          >
            {paused ? "Play All" : "Pause All"}
          </Button>
        </Toolbar>

        <Divider sx={{ borderColor: "rgba(255,255,255,.06)" }} />

        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          textColor="inherit"
          TabIndicatorProps={{ style: { background: "#2ee6ff" } }}
          sx={{
            px: 2,
            "& .MuiTab-root": { textTransform: "none", fontWeight: 700 },
          }}
        >
          <Tab label="Dashboard" />
          <Tab label="Analysis" />
          <Tab label="Global Logs" />
          <Tab label="Settings" />
        </Tabs>
      </AppBar>

      <Box sx={{ p: 2, maxWidth: 1900, mx: "auto" }}>
        {tab === 0 && (
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "1fr",
                md: "repeat(2, minmax(420px, 1fr))",
                xl: "repeat(3, minmax(520px, 1fr))",
              },
              gap: 4,
              py: 3,
              minHeight: "78vh",
            }}
          >
            {zones.map((z) => (
              <Card
                key={z}
                elevation={8}
                onClick={() => setOpenZone(z)} // make whole card clickable
                sx={{
                  cursor: "pointer",
                  bgcolor: "rgba(17,31,61,0.9)",
                  border: "1.5px solid rgba(46,230,255,0.16)",
                  borderRadius: 4,
                  overflow: "hidden",
                  boxShadow: "0 10px 36px rgba(0,0,0,0.35)",
                  transition: "transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s",
                  "&:hover": {
                    transform: "translateY(-6px)",
                    boxShadow: "0 16px 48px rgba(46,230,255,0.22)",
                    borderColor: "#2ee6ff",
                  },
                }}
              >
                <CardHeader
                  title={
                    <Typography sx={{ fontWeight: 900, color: "#2ee6ff", fontSize: 21, letterSpacing: 0.5 }}>
                      Zone {z}
                    </Typography>
                  }
                  subheader={
                    <Typography variant="caption" sx={{ color: "rgba(231,238,252,.78)", fontWeight: 700 }}>
                      Live annotated
                    </Typography>
                  }
                  sx={{ py: 1.3, px: 2 }}
                />
                <Box sx={{ position: "relative", borderRadius: 3, overflow: "hidden", minHeight: "320px" }}>
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      zIndex: 2,
                      pointerEvents: "none",
                      borderRadius: 3,
                      border: "2px solid rgba(46,230,255,0.18)",
                      boxShadow: "0 0 0 2px rgba(46,230,255,0.08)",
                    }}
                  />
                  <CardMedia
                    component="img"
                    image={zoneSrc(z)}
                    alt={`Zone ${z}`}
                    onError={() => {
                      // If MJPEG stream fails once, force fallback to frame polling
                      setImgErr((m) => ({ ...m, [z]: true }));
                      setStreamDisabled((m) => ({ ...m, [z]: true }));
                    }}
                    onLoad={() => setImgErr((m) => ({ ...m, [z]: false }))}
                    sx={{
                      width: "100%",
                      height: { xs: "340px", md: "380px", xl: "420px" },
                      aspectRatio: "16/9",
                      objectFit: "cover",
                      cursor: "pointer",
                      bgcolor: "#050a14",
                      filter: "brightness(1.06) contrast(1.08)",
                      transition: "filter 0.2s",
                      pointerEvents: "none", // card handles click
                    }}
                  />
                  {imgErr[z] && (
                    <Box
                      sx={{
                        position: "absolute",
                        inset: 0,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        bgcolor: "rgba(0,0,0,.55)",
                        color: "#e7eefc",
                        fontWeight: 900,
                        fontSize: 22,
                      }}
                    >
                      Switching to fallback stream...
                    </Box>
                  )}
                </Box>
              </Card>
            ))}
          </Box>
        )}

        {tab === 1 && (
          <Stack spacing={2} sx={{ maxWidth: 940 }}>
            <Typography variant="h6" sx={{ fontWeight: 900, color: "#2ee6ff" }}>
              Analysis
            </Typography>

            <Box
              sx={{
                p: 2.2,
                borderRadius: 3,
                bgcolor: "rgba(15,27,51,.58)",
                border: "1px solid rgba(46,230,255,.14)",
                boxShadow: "0 10px 36px rgba(0,0,0,0.22)",
              }}
            >
              <Typography sx={{ fontWeight: 800, mb: 1 }}>Alerts by zone</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {Object.entries(analysis.alerts_by_zone || {}).map(([k, v]) => (
                  <Chip
                    key={k}
                    label={`${k}: ${v}`}
                    sx={{
                      bgcolor: "rgba(46,230,255,.14)",
                      color: "#e7eefc",
                      border: "1px solid rgba(46,230,255,.22)",
                      fontWeight: 900,
                    }}
                  />
                ))}
              </Stack>
            </Box>

            <Box
              sx={{
                p: 2.2,
                borderRadius: 3,
                bgcolor: "rgba(15,27,51,.58)",
                border: "1px solid rgba(255,91,110,.16)",
                boxShadow: "0 10px 36px rgba(0,0,0,0.22)",
              }}
            >
              <Typography sx={{ fontWeight: 800, mb: 1 }}>Alerts by label</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {Object.entries(analysis.alerts_by_label || {}).map(([k, v]) => (
                  <Chip
                    key={k}
                    label={`${k}: ${v}`}
                    sx={{
                      bgcolor: "rgba(255,91,110,.14)",
                      color: "#e7eefc",
                      border: "1px solid rgba(255,91,110,.24)",
                      fontWeight: 900,
                    }}
                  />
                ))}
              </Stack>
            </Box>
          </Stack>
        )}

        {tab === 2 && (
          <Box sx={{ p: 2, borderRadius: 3, bgcolor: "rgba(15,27,51,.58)", border: "1px solid rgba(255,255,255,.07)" }}>
            <Typography variant="h6" sx={{ fontWeight: 900, color: "#2ee6ff", mb: 1 }}>
              Global Logs
            </Typography>
            <Stack spacing={1}>
              {(alerts.items || []).map((a, idx) => (
                <Box
                  key={idx}
                  sx={{
                    p: 1.1,
                    borderRadius: 2,
                    bgcolor: "rgba(17,31,61,.7)",
                    border: "1px solid rgba(255,255,255,.07)",
                    display: "grid",
                    gridTemplateColumns: "160px 110px 140px 1fr 80px",
                    gap: 1,
                    alignItems: "center",
                  }}
                >
                  <Typography variant="caption" sx={{ color: "rgba(231,238,252,.65)" }}>
                    {a.ts}
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 800 }}>
                    {a.zone}
                  </Typography>
                  <Typography variant="caption" sx={{ color: "#ff5b6e", fontWeight: 900 }}>
                    {a.label}
                  </Typography>
                  <Typography variant="caption" sx={{ color: "rgba(231,238,252,.85)" }}>
                    {a.msg}
                  </Typography>
                  <Typography variant="caption" sx={{ textAlign: "right", color: "rgba(231,238,252,.65)" }}>
                    {(a.conf ?? 0).toFixed(2)}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Box>
        )}

        {tab === 3 && (
          <Box sx={{ p: 2, borderRadius: 3, bgcolor: "rgba(15,27,51,.58)", border: "1px solid rgba(255,255,255,.07)" }}>
            <Typography variant="h6" sx={{ fontWeight: 900, color: "#2ee6ff" }}>
              Settings
            </Typography>
            <Typography sx={{ color: "rgba(231,238,252,.7)", mt: 1 }}>
              Backend: {API}. Model is loaded server-side (best.pt fallback yolov8n.pt).
            </Typography>
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
            bgcolor: "rgba(15,27,51,0.98)",
            borderRadius: { xs: 0, md: 4 },
            boxShadow: "0 12px 64px 0 rgba(46,230,255,0.16)",
            border: "2px solid #2ee6ff",
            p: 0,
            overflow: "hidden",
            width: { xs: "100vw", md: "96vw" },
            height: { xs: "100vh", md: "92vh" },
            maxWidth: "none",
            m: { xs: 0, md: 2 },
          },
        }}
      >
        <Box sx={{ color: "#e7eefc", height: "100%", display: "flex", flexDirection: "column" }}>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ p: 2, pb: 1 }}>
            <Typography sx={{ fontWeight: 900, color: "#2ee6ff", fontSize: 24, letterSpacing: 0.5 }}>
              Zone {openZone} — Zoom & Pan
            </Typography>
            <Stack direction="row" spacing={1} alignItems="center">
              <Button
                variant={modalPaused ? "outlined" : "contained"}
                startIcon={modalPaused ? <PlayArrowIcon /> : <PauseIcon />}
                onClick={() => setModalPaused((p) => !p)}
                sx={{
                  borderColor: "rgba(46,230,255,.35)",
                  bgcolor: modalPaused ? "transparent" : "#2ee6ff",
                  color: modalPaused ? "#2ee6ff" : "#0b1220",
                  fontWeight: 900,
                }}
              >
                {modalPaused ? "Play" : "Pause"}
              </Button>
              <IconButton onClick={() => setOpenZone(null)} sx={{ color: "#e7eefc", fontSize: 30 }}>
                <CloseIcon fontSize="inherit" />
              </IconButton>
            </Stack>
          </Stack>
          <Divider sx={{ borderColor: "rgba(46,230,255,0.18)" }} />

          <Box
            sx={{
              p: 3,
              pt: 2,
              flex: 1,
              minHeight: 0,
              height: "100%",
              display: "grid",
              gap: 2,
              gridTemplateColumns: { xs: "1fr", lg: "1fr 320px" },
              alignItems: "stretch",
            }}
          >
            <Box
              sx={{
                // IMPORTANT: give the zoom area real, stable height so the image can't collapse to 0px
                flex: 1,
                minHeight: 0,
                minWidth: 0,
                height: { xs: "60vh", lg: "100%" },
                display: "flex",
                flexDirection: "column",
              }}
            >
              <Box
                sx={{
                  position: "relative",
                  flex: 1,
                  minHeight: 0,
                  borderRadius: 2,
                  overflow: "hidden",
                  background: "#050a14",
                  border: "2.2px solid #2ee6ff",
                  boxShadow: "0 4px 32px 0 rgba(46,230,255,0.12)",
                  // Prevent page/Modal scroll while interacting with zoom/pinch
                  touchAction: "none",
                  overscrollBehavior: "contain",
                }}
                ref={modalVideoBoxRef}
                onWheelCapture={(e) => e.stopPropagation()}
                onTouchMoveCapture={(e) => e.stopPropagation()}
              >
                {openZone && modalBackendOk === false && (
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      zIndex: 6,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      bgcolor: "rgba(0,0,0,.60)",
                      color: "#e7eefc",
                      fontWeight: 900,
                      p: 2,
                      textAlign: "center",
                    }}
                  >
                    Backend is not serving frames for Zone {openZone}. Open{" "}
                    <a
                      href={`${API}/api/zones/${openZone}/frame.jpg`}
                      target="_blank"
                      rel="noreferrer"
                      style={{ color: "#2ee6ff", textDecoration: "underline" }}
                    >
                      {API}/api/zones/{openZone}/frame.jpg
                    </a>
                  </Box>
                )}

                <TransformWrapper
                  wheel={{ step: 0.15 }}
                  doubleClick={{ disabled: true }}
                  panning={{ disabled: false, velocityDisabled: true }}
                  pinch={{ disabled: false }}
                  minScale={1}
                  centerOnInit
                  limitToBounds={false}
                >
                  <TransformComponent
                    wrapperStyle={{
                      width: "100%",
                      height: "100%",
                      touchAction: "none",
                      overscrollBehavior: "contain",
                      cursor: "grab",
                      background: "#050a14",
                    }}
                    contentStyle={{
                      width: "100%",
                      height: "100%",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      touchAction: "none",
                      background: "#050a14",
                    }}
                  >
                    {/* Fallback frame (always available; shown until MJPEG first frame arrives) */}
                    <img
                      src={openZone ? modalBlobUrl : ""}
                      alt={`Zone ${openZone} Fallback`}
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        background: "#050a14",
                        userSelect: "none",
                        pointerEvents: "none",
                        display: modalBlobUrl ? "block" : "none",
                      }}
                    />

                    {/* MJPEG stream (preferred for smoothness) */}
                    <img
                      src={openZone && modalUseMjpeg && !modalPaused ? `${API}/api/zones/${openZone}/stream.mjpg` : ""}
                      alt={`Zone ${openZone} Large`}
                      onLoad={() => setModalMjpegOk(true)}
                      onError={() => {
                        setModalUseMjpeg(false);
                        setModalMjpegOk(false);
                      }}
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        background: "#050a14",
                        userSelect: "none",
                        pointerEvents: "none",
                        display: openZone && modalUseMjpeg && !modalPaused ? "block" : "none",
                      }}
                    />
                  </TransformComponent>
                </TransformWrapper>
              </Box>

              <Typography variant="caption" sx={{ display: "block", mt: 2, color: "rgba(231,238,252,.75)", fontWeight: 800, fontSize: 16 }}>
                Use wheel / pinch to zoom. Drag to pan.
              </Typography>
            </Box>

            <Box
              sx={{
                minHeight: 0,
                borderRadius: 3,
                bgcolor: "rgba(15,27,51,.58)",
                border: "1px solid rgba(46,230,255,.14)",
                p: 2,
                overflow: "auto",
              }}
            >
              <Typography sx={{ fontWeight: 950, color: "#2ee6ff", mb: 1.2, fontSize: 18 }}>
                Live Analysis
              </Typography>

              <Typography sx={{ color: "rgba(231,238,252,.75)", fontWeight: 900, mb: 1 }}>
                Objects detected now: {modalDetections.count ?? 0}
              </Typography>

              <Typography sx={{ fontWeight: 900, mb: 1 }}>By label (recent)</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 2 }}>
                {Object.entries(modalAnalysis.alerts_by_label || {}).map(([k, v]) => (
                  <Chip
                    key={k}
                    label={`${k}: ${v}`}
                    sx={{ bgcolor: "rgba(0,220,0,.14)", border: "1px solid rgba(0,220,0,.28)", color: "#e7eefc", fontWeight: 900 }}
                  />
                ))}
              </Stack>

              <Typography sx={{ fontWeight: 900, mb: 1 }}>Detected objects (live)</Typography>
              <Stack spacing={1} sx={{ mb: 2 }}>
                {(modalDetections.items || []).slice(0, 12).map((d, idx) => {
                  const isEmergency = String(d.label || "").toLowerCase().includes("drown") || String(d.label || "").toLowerCase().includes("emerg");
                  return (
                    <Box
                      key={idx}
                      sx={{
                        p: 1,
                        borderRadius: 2,
                        bgcolor: "rgba(17,31,61,.7)",
                        border: `2px solid ${isEmergency ? "rgba(255,165,0,.55)" : "rgba(0,220,0,.55)"}`,
                        display: "grid",
                        gridTemplateColumns: "1fr 70px",
                        gap: 1,
                        alignItems: "center",
                      }}
                    >
                      <Typography variant="caption" sx={{ fontWeight: 950, color: isEmergency ? "#ffa500" : "#00dc00" }}>
                        {d.label}
                      </Typography>
                      <Typography variant="caption" sx={{ textAlign: "right", fontWeight: 950, color: "rgba(231,238,252,.88)" }}>
                        {(d.conf ?? 0).toFixed(2)}
                      </Typography>
                    </Box>
                  );
                })}
              </Stack>

              <Typography sx={{ fontWeight: 900, mb: 1 }}>Latest events</Typography>
              <Stack spacing={1}>
                {(modalAlerts.items || []).slice(0, 12).map((a, idx) => (
                  <Box
                    key={idx}
                    sx={{
                      p: 1,
                      borderRadius: 2,
                      bgcolor: "rgba(17,31,61,.7)",
                      border: "1px solid rgba(255,255,255,.07)",
                      display: "grid",
                      gridTemplateColumns: "1fr 70px",
                      gap: 1,
                      alignItems: "center",
                    }}
                  >
                    <Box>
                      <Typography variant="caption" sx={{ color: "rgba(231,238,252,.70)", fontWeight: 800, display: "block" }}>
                        {a.ts}
                      </Typography>
                      <Typography variant="caption" sx={{ fontWeight: 950, color: "#2ee6ff" }}>
                        {a.label}
                      </Typography>
                    </Box>
                    <Typography variant="caption" sx={{ textAlign: "right", fontWeight: 950, color: "rgba(231,238,252,.88)" }}>
                      {(a.conf ?? 0).toFixed(2)}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Box>
          </Box>
        </Box>
      </Dialog>
    </Box>
  );
}
