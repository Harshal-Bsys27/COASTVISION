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
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import PauseIcon from "@mui/icons-material/Pause";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
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

  const height = { xs: "340px", md: "380px", xl: "420px" };

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

      {/* Live detection overlay */}
      <Box
        sx={{
          position: "absolute",
          left: 12,
          bottom: 12,
          zIndex: 3,
          px: 1.2,
          py: 0.6,
          borderRadius: 2,
          bgcolor: "rgba(0,0,0,.55)",
          border: `1px solid ${emergency ? "rgba(255,165,0,.55)" : "rgba(46,230,255,.30)"}`,
          color: emergency ? "#ffa500" : "#e7eefc",
          fontWeight: 900,
          fontSize: 13,
          pointerEvents: "none",
        }}
      >
        Detections: {det.count ?? 0}
      </Box>
    </Box>
  );
}

export default function App() {
  const [tab, setTab] = useState(0);
  const [paused, setPaused] = useState(false);
  const [openZone, setOpenZone] = useState(null);

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

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "linear-gradient(135deg, #0b1220 60%, #1a233a 100%)", color: "#e7eefc", backgroundAttachment: "fixed" }}>
      <AppBar position="sticky" elevation={0} sx={{ bgcolor: "rgba(15,27,51,.85)", backdropFilter: "blur(16px)", boxShadow: "0 4px 32px 0 rgba(46,230,255,0.08)", borderBottom: "1.5px solid rgba(46,230,255,0.10)" }}>
        <Toolbar sx={{ gap: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: 0.5 }}>CoastVision</Typography>

          <Chip
            label={backendOk === false ? "Backend: Offline" : backendOk === true ? "Backend: Online" : "Backend: …"}
            size="small"
            sx={{ bgcolor: backendOk === false ? "rgba(255,90,90,.14)" : "rgba(46,230,255,.12)", color: backendOk === false ? "rgba(255,120,120,1)" : "#2ee6ff", fontWeight: 800 }}
          />
          <Chip label={`Device: ${health?.device ?? "?"}`} size="small" sx={{ bgcolor: "rgba(46,230,255,.08)", color: "rgba(231,238,252,.92)", fontWeight: 800 }} />
          <Chip label={`Alerts: ${analysis.alerts_total ?? 0}`} size="small" sx={{ bgcolor: "rgba(46,230,255,.12)", color: "#2ee6ff" }} />

          <Box sx={{ flex: 1 }} />

          <Button
            variant="outlined"
            onClick={() => fetch(`${API}/api/zones/reload`, { method: "POST" }).catch(() => {})}
            sx={{ borderColor: "rgba(46,230,255,.22)", color: "rgba(231,238,252,.9)", fontWeight: 900 }}
          >
            Reload Zones
          </Button>

          <Button
            variant={paused ? "outlined" : "contained"}
            startIcon={paused ? <PlayArrowIcon /> : <PauseIcon />}
            onClick={() => setPaused((p) => !p)}
            sx={{ borderColor: "rgba(46,230,255,.35)", bgcolor: paused ? "transparent" : "#2ee6ff", color: paused ? "#2ee6ff" : "#0b1220", fontWeight: 800 }}
          >
            {paused ? "Play All" : "Pause All"}
          </Button>
        </Toolbar>

        <Divider sx={{ borderColor: "rgba(255,255,255,.06)" }} />

        <Tabs value={tab} onChange={(_, v) => setTab(v)} textColor="inherit" TabIndicatorProps={{ style: { background: "#2ee6ff" } }} sx={{ px: 2, "& .MuiTab-root": { textTransform: "none", fontWeight: 700 } }}>
          <Tab label="Dashboard" />
          <Tab label="Analysis" />
          <Tab label="Global Logs" />
          <Tab label="Settings" />
        </Tabs>
      </AppBar>

      <Box sx={{ p: 2, maxWidth: 1900, mx: "auto" }}>
        {tab === 0 && (
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(420px, 1fr))", xl: "repeat(3, minmax(520px, 1fr))" }, gap: 4, py: 3, minHeight: "78vh" }}>
            {zones.length === 0 ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, alignItems: "center", justifyContent: "center", textAlign: "center", gridColumn: "1/-1", minHeight: "50vh" }}>
                <Typography sx={{ fontWeight: 900, fontSize: 20, color: "#2ee6ff" }}>No zones detected</Typography>
                <Typography sx={{ color: "rgba(231,238,252,.78)" }}>Put MP4 files named zone1.mp4 … zoneN.mp4 into the videos folder, then click Reload Zones.</Typography>
              </Box>
            ) : (
              zones.map((z) => {
                const meta = zoneMeta.get(z) || {};
                const exists = !!meta.exists;
                const showError = typeof meta.last_error === "string" && meta.last_error.length > 0;

                return (
                  <Card
                    key={z}
                    elevation={8}
                    onClick={() => setOpenZone(z)}
                    sx={{
                      cursor: "pointer",
                      bgcolor: "rgba(17,31,61,0.9)",
                      border: "1.5px solid rgba(46,230,255,0.16)",
                      borderRadius: 4,
                      overflow: "hidden",
                      boxShadow: "0 10px 36px rgba(0,0,0,0.35)",
                    }}
                  >
                    <CardHeader
                      title={<Typography sx={{ fontWeight: 900, color: "#2ee6ff", fontSize: 21, letterSpacing: 0.5 }}>Zone {z}</Typography>}
                      subheader={
                        <Typography variant="caption" sx={{ color: "rgba(231,238,252,.78)", fontWeight: 700 }}>
                          {exists ? "Live annotated" : "Video missing"}
                        </Typography>
                      }
                      sx={{ py: 1.3, px: 2 }}
                    />

                    <Box sx={{ position: "relative", borderRadius: 3, overflow: "hidden", minHeight: "320px" }}>
                      <Box sx={{ position: "absolute", inset: 0, zIndex: 2, pointerEvents: "none", borderRadius: 3, border: "2px solid rgba(46,230,255,0.18)" }} />
                      <ZoneCardPlayback zid={z} paused={paused} />

                      {(!exists || showError) && (
                        <Box
                          sx={{
                            position: "absolute",
                            inset: 0,
                            zIndex: 4,
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            justifyContent: "center",
                            bgcolor: "rgba(0,0,0,.55)",
                            color: "#e7eefc",
                            fontWeight: 900,
                            px: 2,
                            textAlign: "center",
                          }}
                        >
                          <Typography sx={{ fontWeight: 900, fontSize: 20 }}>{exists ? "Stream issue" : "Video file missing"}</Typography>
                          {showError && <Typography variant="caption" sx={{ color: "rgba(231,238,252,.74)" }}>{meta.last_error}</Typography>}
                        </Box>
                      )}
                    </Box>
                  </Card>
                );
              })
            )}
          </Box>
        )}

        {tab === 1 && (
          <Stack spacing={2} sx={{ maxWidth: 940 }}>
            <Typography variant="h6" sx={{ fontWeight: 900, color: "#2ee6ff" }}>Analysis</Typography>

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
            <Typography variant="h6" sx={{ fontWeight: 900, color: "#2ee6ff", mb: 1 }}>Global Logs</Typography>
            <Stack spacing={1}>
              {(alerts.items || []).map((a, idx) => (
                <Box key={idx} sx={{ p: 1.1, borderRadius: 2, bgcolor: "rgba(17,31,61,.7)", border: "1px solid rgba(255,255,255,.07)", display: "grid", gridTemplateColumns: "160px 110px 140px 1fr 80px", gap: 1, alignItems: "center" }}>
                  <Typography variant="caption" sx={{ color: "rgba(231,238,252,.65)" }}>{a.ts}</Typography>
                  <Typography variant="caption" sx={{ fontWeight: 800 }}>{`Zone ${a.zone}`}</Typography>
                  <Typography variant="caption" sx={{ color: "#ff5b6e", fontWeight: 900 }}>{a.label}</Typography>
                  <Typography variant="caption" sx={{ color: "rgba(231,238,252,.85)" }}>{a.msg}</Typography>
                  <Typography variant="caption" sx={{ textAlign: "right", color: "rgba(231,238,252,.65)" }}>{(a.conf ?? 0).toFixed(2)}</Typography>
                </Box>
              ))}
            </Stack>
          </Box>
        )}

        {tab === 3 && (
          <Box sx={{ p: 2, borderRadius: 3, bgcolor: "rgba(15,27,51,.58)", border: "1px solid rgba(255,255,255,.07)" }}>
            <Typography variant="h6" sx={{ fontWeight: 900, color: "#2ee6ff" }}>Settings</Typography>
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
            <Typography sx={{ fontWeight: 900, color: "#2ee6ff", fontSize: 24, letterSpacing: 0.5 }}>Zone {openZone} — Zoom & Pan</Typography>
            <Stack direction="row" spacing={1} alignItems="center">
              <Button
                variant={modalPaused ? "outlined" : "contained"}
                startIcon={modalPaused ? <PlayArrowIcon /> : <PauseIcon />}
                onClick={() => setModalPaused((p) => !p)}
                sx={{ borderColor: "rgba(46,230,255,.35)", bgcolor: modalPaused ? "transparent" : "#2ee6ff", color: modalPaused ? "#2ee6ff" : "#0b1220", fontWeight: 900 }}
              >
                {modalPaused ? "Play" : "Pause"}
              </Button>
              <IconButton onClick={() => setOpenZone(null)} sx={{ color: "#e7eefc", fontSize: 30 }}>
                <CloseIcon fontSize="inherit" />
              </IconButton>
            </Stack>
          </Stack>

          <Divider sx={{ borderColor: "rgba(46,230,255,0.18)" }} />

          <Box sx={{ p: 3, pt: 2, flex: 1, minHeight: 0, height: "100%", display: "grid", gap: 2, gridTemplateColumns: { xs: "1fr", lg: "1fr 320px" }, alignItems: "stretch" }}>
            <Box sx={{ flex: 1, minHeight: 0, minWidth: 0, height: { xs: "60vh", lg: "100%" }, display: "flex", flexDirection: "column" }}>
              <Box
                ref={modalVideoBoxRef}
                sx={{
                  position: "relative",
                  flex: 1,
                  minHeight: 0,
                  borderRadius: 2,
                  overflow: "hidden",
                  background: "#050a14",
                  border: "2.2px solid #2ee6ff",
                  boxShadow: "0 4px 32px 0 rgba(46,230,255,0.12)",
                  touchAction: "none",
                  overscrollBehavior: "contain",
                }}
              >
                <TransformWrapper wheel={{ step: 0.15 }} doubleClick={{ disabled: true }} panning={{ disabled: false, velocityDisabled: true }} pinch={{ disabled: false }} minScale={1} centerOnInit limitToBounds={false}>
                  <TransformComponent
                    wrapperStyle={{ width: "100%", height: "100%", touchAction: "none", overscrollBehavior: "contain", cursor: "grab", background: "#050a14" }}
                    contentStyle={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", touchAction: "none", background: "#050a14" }}
                  >
                    {/* Fallback frame (always available) */}
                    <img
                      src={openZone ? modalBlobUrl : ""}
                      alt={`Zone ${openZone} Fallback`}
                      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", background: "#050a14", userSelect: "none", pointerEvents: "none", display: modalBlobUrl ? "block" : "none" }}
                    />

                    {/* MJPEG stream (preferred) */}
                    <img
                      src={openZone && modalUseMjpeg && !modalPaused ? `${API}/api/zones/${openZone}/stream.mjpg` : ""}
                      alt={`Zone ${openZone} Stream`}
                      onLoad={() => setModalMjpegOk(true)}
                      onError={() => {
                        setModalUseMjpeg(false);
                        setModalMjpegOk(false);
                      }}
                      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", background: "#050a14", userSelect: "none", pointerEvents: "none", display: openZone && modalUseMjpeg && !modalPaused ? "block" : "none" }}
                    />
                  </TransformComponent>
                </TransformWrapper>
              </Box>

              <Typography variant="caption" sx={{ display: "block", mt: 2, color: "rgba(231,238,252,.75)", fontWeight: 800, fontSize: 16 }}>
                Use wheel / pinch to zoom. Drag to pan.
              </Typography>
            </Box>

            <Box sx={{ minHeight: 0, borderRadius: 3, bgcolor: "rgba(15,27,51,.58)", border: "1px solid rgba(46,230,255,.14)", p: 2, overflow: "auto" }}>
              <Typography sx={{ fontWeight: 950, color: "#2ee6ff", mb: 1.2, fontSize: 18 }}>Live Analysis</Typography>
              <Typography sx={{ color: "rgba(231,238,252,.75)", fontWeight: 900, mb: 1 }}>Objects detected now: {modalDetections.count ?? 0}</Typography>

              <Typography sx={{ fontWeight: 900, mb: 1 }}>Latest events</Typography>
              <Stack spacing={1}>
                {(modalAlerts.items || []).slice(0, 12).map((a, idx) => (
                  <Box key={idx} sx={{ p: 1, borderRadius: 2, bgcolor: "rgba(17,31,61,.7)", border: "1px solid rgba(255,255,255,.07)", display: "grid", gridTemplateColumns: "1fr 70px", gap: 1, alignItems: "center" }}>
                    <Box>
                      <Typography variant="caption" sx={{ color: "rgba(231,238,252,.70)", fontWeight: 800, display: "block" }}>{a.ts}</Typography>
                      <Typography variant="caption" sx={{ fontWeight: 950, color: "#2ee6ff" }}>{a.label}</Typography>
                    </Box>
                    <Typography variant="caption" sx={{ textAlign: "right", fontWeight: 950, color: "rgba(231,238,252,.88)" }}>{(a.conf ?? 0).toFixed(2)}</Typography>
                  </Box>
                ))}
              </Stack>

              {/* ...you can re-add your detailed chips/panels here... */}
            </Box>
          </Box>
        </Box>
      </Dialog>
    </Box>
  );
}
