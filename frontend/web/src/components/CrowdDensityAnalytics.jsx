import React, { useEffect, useState } from "react";
import { Box, Button, Chip, LinearProgress, Stack, TextField, Typography } from "@mui/material";
import WavesIcon from "@mui/icons-material/Waves";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";

export default function CrowdDensityAnalytics({ api, zones, zoneNames }) {
  const [crowdData, setCrowdData] = useState(null);
  const [crowdAlerts, setCrowdAlerts] = useState([]);
  const [editingZone, setEditingZone] = useState(null);
  const [newThreshold, setNewThreshold] = useState("");
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCrowdData = async () => {
      try {
        const [statusRes, alertsRes] = await Promise.all([
          fetch(`${api}/api/analytics/crowd-status`),
          fetch(`${api}/api/analytics/crowd-alerts?limit=50`)
        ]);
        
        const statusData = await statusRes.json();
        const alertsData = await alertsRes.json();
        
        setCrowdData(statusData);
        setCrowdAlerts(alertsData.alerts || []);
      } catch (e) {
        console.error("Failed to fetch crowd data:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchCrowdData();
    const interval = setInterval(fetchCrowdData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [api]);

  const handleThresholdChange = async (zid) => {
    if (!newThreshold || isNaN(newThreshold)) return;
    
    setSaving(true);
    try {
      const res = await fetch(`${api}/api/zones/${zid}/crowd-threshold`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold: parseInt(newThreshold) })
      });
      
      if (res.ok) {
        setEditingZone(null);
        setNewThreshold("");
        // Refresh data
        const crowdRes = await fetch(`${api}/api/analytics/crowd-status`);
        const newData = await crowdRes.json();
        setCrowdData(newData);
      }
    } catch (e) {
      console.error("Failed to update threshold:", e);
    } finally {
      setSaving(false);
    }
  };

  const getHeatmapColor = (percent) => {
    if (percent < 30) return { bg: "rgba(52, 211, 153, 0.15)", border: "#34d399", text: "#34d399", label: "Safe" };
    if (percent < 60) return { bg: "rgba(34, 211, 238, 0.15)", border: "#22d3ee", text: "#22d3ee", label: "Normal" };
    if (percent < 85) return { bg: "rgba(245, 158, 11, 0.15)", border: "#f59e0b", text: "#f59e0b", label: "Caution" };
    return { bg: "rgba(255, 82, 82, 0.2)", border: "#ff5252", text: "#ff5252", label: "Critical" };
  };

  if (loading || !crowdData) {
    return (
      <Box sx={{ py: 6, textAlign: "center" }}>
        <LinearProgress sx={{ bgcolor: "rgba(255,255,255,0.05)" }} />
        <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 13, mt: 2 }}>Loading crowd data...</Typography>
      </Box>
    );
  }

  const { zones: zonesStatus, crowded_zones_count, overall_safety } = crowdData;
  const zoneArray = Object.entries(zonesStatus);

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: "1fr", gap: 3 }}>
      
      {/* ── Overall Safety Status ── */}
      <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
          <WavesIcon sx={{ fontSize: 28, color: overall_safety === "safe" ? "#34d399" : overall_safety === "warning" ? "#f59e0b" : "#ff5252" }} />
          <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>Overall Safety Status</Typography>
        </Box>
        
        <Stack spacing={2}>
          <Box sx={{
            p: 3,
            borderRadius: 2,
            bgcolor: overall_safety === "safe" ? "rgba(52,211,153,0.1)" : overall_safety === "warning" ? "rgba(245,158,11,0.1)" : "rgba(255,82,82,0.1)",
            border: `1px solid ${overall_safety === "safe" ? "rgba(52,211,153,0.2)" : overall_safety === "warning" ? "rgba(245,158,11,0.2)" : "rgba(255,82,82,0.2)"}`
          }}>
            <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.6)", fontWeight: 600, mb: 1, textTransform: "uppercase" }}>Status</Typography>
            <Typography sx={{
              fontSize: 32,
              fontWeight: 900,
              color: overall_safety === "safe" ? "#34d399" : overall_safety === "warning" ? "#f59e0b" : "#ff5252",
              textTransform: "uppercase"
            }}>
              {overall_safety}
            </Typography>
          </Box>
          
          <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 2 }}>
            <Box sx={{ p: 2.5, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
              <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 12, fontWeight: 600, mb: 0.5 }}>Crowded Zones</Typography>
              <Typography sx={{ fontSize: 28, fontWeight: 900, color: crowded_zones_count > 0 ? "#ff5252" : "#34d399" }}>{crowded_zones_count}</Typography>
            </Box>
            <Box sx={{ p: 2.5, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
              <Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 12, fontWeight: 600, mb: 0.5 }}>Total Zones</Typography>
              <Typography sx={{ fontSize: 28, fontWeight: 900, color: "#2dd4bf" }}>{zoneArray.length}</Typography>
            </Box>
          </Box>
        </Stack>
      </Box>

      {/* ── CROWD DENSITY ZONES ── */}
      <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#060c12", border: "2px solid rgba(45,212,191,0.15)", boxShadow: "0 8px 32px rgba(0,0,0,0.4)" }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Box sx={{ width: 12, height: 12, borderRadius: "50%", bgcolor: "#2dd4bf", boxShadow: "0 0 12px #2dd4bf80", animation: "pulse 2s infinite" }} />
            <Typography sx={{ fontWeight: 900, fontSize: 26, color: "#fff" }}>Crowd Density</Typography>
          </Box>
          <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.45)", fontWeight: 600 }}>Live zone monitoring</Typography>
        </Box>

        {/* Zone Cards Grid */}
        <Box sx={{
          display: "grid",
          gridTemplateColumns: { xs: "repeat(2, 1fr)", sm: "repeat(3, 1fr)", md: "repeat(4, 1fr)", lg: "repeat(5, 1fr)" },
          gap: 2.5,
          mb: 3
        }}>
          {zoneArray.map(([zid, status]) => {
            const percent = Math.min(100, (status.crowding_level || 0));
            let cardBg = "#1a3a3a", borderColor = "#34d399", badgeColor = "#34d399", badgeLabel = "✅ Safe"; // Safe
            if (percent >= 30) { cardBg = "#1a3a4a"; borderColor = "#22d3ee"; badgeColor = "#22d3ee"; badgeLabel = "📊 Normal"; }
            if (percent >= 60) { cardBg = "#3a3a1a"; borderColor = "#f59e0b"; badgeColor = "#f59e0b"; badgeLabel = "⚠️ Caution"; }
            if (percent >= 85) { cardBg = "#3a1a1a"; borderColor = "#ff5252"; badgeColor = "#ff5252"; badgeLabel = "🔴 Critical"; }
            
            const isEditing = editingZone === parseInt(zid);

            return (
              <Box
                key={zid}
                sx={{
                  p: 2.5,
                  borderRadius: 3,
                  bgcolor: cardBg,
                  border: `2px solid ${borderColor}`,
                  boxShadow: `0 0 20px ${borderColor}20, inset 0 1px 2px rgba(255,255,255,0.05)`,
                  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                  position: "relative",
                  overflow: "hidden",
                  "&:before": {
                    content: '""',
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: "2px",
                    background: `linear-gradient(90deg, transparent, ${borderColor}, transparent)`,
                    animation: "shimmer 2s infinite",
                  },
                  "&:hover": {
                    boxShadow: `0 0 30px ${borderColor}40, inset 0 1px 2px rgba(255,255,255,0.1)`,
                    transform: "translateY(-2px)",
                  },
                  "@keyframes shimmer": {
                    "0%, 100%": { opacity: 0.5 },
                    "50%": { opacity: 1 },
                  },
                }}
              >
                {/* Status Badge */}
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1.5 }}>
                  <Chip
                    label={badgeLabel}
                    size="small"
                    sx={{
                      bgcolor: `${badgeColor}20`,
                      color: badgeColor,
                      fontWeight: 800,
                      height: 28,
                      fontSize: 11,
                      border: `1px solid ${badgeColor}`,
                      borderRadius: 1.5
                    }}
                  />
                  <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.4)", fontWeight: 600 }}>{percent.toFixed(0)}%</Typography>
                </Box>

                {/* Zone Name */}
                <Typography sx={{ fontSize: 14, fontWeight: 800, color: "#fff", mb: 1.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {status.zone_name}
                </Typography>

                {/* Person Count - Large Display */}
                <Box sx={{ mb: 2, p: 2, borderRadius: 2, bgcolor: "rgba(0,0,0,0.3)", textAlign: "center", border: `1px solid ${borderColor}30` }}>
                  <Typography sx={{ fontSize: 28, fontWeight: 900, color: borderColor, textShadow: `0 0 12px ${borderColor}40` }}>
                    {status.person_count}
                  </Typography>
                  <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 600 }}>People Detected</Typography>
                </Box>

                {/* Progress Bar */}
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
                    <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.5)", fontWeight: 600 }}>Crowding Level</Typography>
                    <Typography sx={{ fontSize: 10, color: borderColor, fontWeight: 700 }}>{percent.toFixed(0)}% / 100%</Typography>
                  </Box>
                  <Box sx={{ width: "100%", height: 6, borderRadius: 3, bgcolor: "rgba(255,255,255,0.08)", border: `1px solid ${borderColor}20`, overflow: "hidden" }}>
                    <Box
                      sx={{
                        height: "100%",
                        width: `${percent}%`,
                        background: `linear-gradient(90deg, ${borderColor}, ${borderColor}80)`,
                        borderRadius: 3,
                        transition: "width 0.4s ease",
                        boxShadow: `0 0 12px ${borderColor}60`
                      }}
                    />
                  </Box>
                </Box>

                {/* Threshold Section */}
                <Box sx={{ mb: 2, p: 1.5, borderRadius: 2, bgcolor: "rgba(0,0,0,0.2)", border: `1px solid ${borderColor}15` }}>
                  <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.4)", fontWeight: 600, mb: 0.5 }}>Threshold Setting</Typography>
                  {isEditing ? (
                    <Box sx={{ display: "flex", gap: 0.75 }}>
                      <TextField
                        autoFocus
                        size="small"
                        type="number"
                        value={newThreshold}
                        onChange={(e) => setNewThreshold(e.target.value)}
                        placeholder={status.threshold.toString()}
                        sx={{
                          flex: 1,
                          "& .MuiOutlinedInput-root": {
                            height: 32,
                            bgcolor: "rgba(0,0,0,0.3)",
                            color: "#fff",
                            fontSize: 12,
                          }
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handleThresholdChange(parseInt(zid));
                          if (e.key === "Escape") setEditingZone(null);
                        }}
                      />
                      <Button
                        size="small"
                        onClick={() => handleThresholdChange(parseInt(zid))}
                        disabled={saving}
                        sx={{ 
                          bgcolor: "#2dd4bf", 
                          color: "#000", 
                          fontWeight: 800, 
                          fontSize: 11, 
                          px: 1.5, 
                          height: 32,
                          transition: "all 0.2s",
                          "&:hover": { bgcolor: "#14b8a6" },
                          "&:disabled": { opacity: 0.6 }
                        }}
                      >
                        Save
                      </Button>
                    </Box>
                  ) : (
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <Box>
                        <Typography sx={{ fontSize: 18, fontWeight: 900, color: borderColor }}>
                          {status.threshold}
                        </Typography>
                        <Typography sx={{ fontSize: 9, color: "rgba(255,255,255,0.4)" }}>people limit</Typography>
                      </Box>
                      <Button
                        size="small"
                        onClick={() => { setEditingZone(parseInt(zid)); setNewThreshold(status.threshold.toString()); }}
                        sx={{
                          color: borderColor,
                          fontSize: 10,
                          fontWeight: 700,
                          border: `1px solid ${borderColor}`,
                          borderRadius: 1.5,
                          px: 1.5,
                          py: 0.75,
                          transition: "all 0.2s",
                          "&:hover": { bgcolor: `${borderColor}15` }
                        }}
                      >
                        ✏️ Edit
                      </Button>
                    </Box>
                  )}
                </Box>
              </Box>
            );
          })}
        </Box>

        {/* Summary Stats */}
        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr 1fr", sm: "repeat(4, 1fr)" }, gap: 2, p: 3, borderRadius: 3, bgcolor: "rgba(45,212,191,0.05)", border: "1px solid rgba(45,212,191,0.12)" }}>
          {[
            { label: "📍 Total Zones", value: zoneArray.length, unit: "zones", color: "#2dd4bf" },
            { label: "👥 Total People", value: zoneArray.reduce((sum, [_, z]) => sum + (z.person_count || 0), 0), unit: "people", color: "#22d3ee" },
            { label: "⚠️ Alert Zones", value: zoneArray.filter(([_, z]) => z.crowding_level >= 60).length, unit: "zones", color: "#f59e0b" },
            { label: "🔴 Critical Zones", value: zoneArray.filter(([_, z]) => z.crowding_level >= 85).length, unit: "zones", color: "#ff5252" }
          ].map((stat) => (
            <Box key={stat.label} sx={{ textAlign: "center", p: 2, borderRadius: 2, bgcolor: "rgba(0,0,0,0.15)", border: `1px solid ${stat.color}15`, transition: "all 0.3s" }}>
              <Typography sx={{ fontSize: 11, color: "rgba(255,255,255,0.5)", fontWeight: 700, mb: 0.75 }}>{stat.label}</Typography>
              <Box sx={{ display: "flex", alignItems: "baseline", justifyContent: "center", gap: 0.5 }}>
                <Typography sx={{ fontSize: 22, fontWeight: 900, color: stat.color }}>
                  {stat.value}
                </Typography>
                <Typography sx={{ fontSize: 10, color: "rgba(255,255,255,0.4)" }}>{stat.unit}</Typography>
              </Box>
            </Box>
          ))}
        </Box>
      </Box>

      {/* ── Crowd Alerts History ── */}
      {crowdAlerts.length > 0 && (
        <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
          <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff", mb: 3 }}>📋 Crowding Incidents</Typography>
          
          <Box sx={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Zone</th>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Person Count</th>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Threshold</th>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Severity</th>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Time</th>
                </tr>
              </thead>
              <tbody>
                {crowdAlerts.slice(0, 30).map((alert, i) => {
                  const severityColor = alert.severity === "low" ? "#f59e0b" : alert.severity === "medium" ? "#ff5252" : "#ff1744";
                  return (
                    <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "12px 16px", fontWeight: 600, color: "#2dd4bf" }}>{alert.zone_name}</td>
                      <td style={{ padding: "12px 16px", color: "#fff", fontWeight: 600 }}>{alert.person_count}</td>
                      <td style={{ padding: "12px 16px", color: "rgba(255,255,255,0.7)" }}>{alert.threshold}</td>
                      <td style={{ padding: "12px 16px" }}>
                        <Chip
                          label={alert.severity.toUpperCase()}
                          size="small"
                          sx={{
                            bgcolor: `${severityColor}20`,
                            color: severityColor,
                            fontWeight: 700,
                            height: 24,
                            fontSize: 10
                          }}
                        />
                      </td>
                      <td style={{ padding: "12px 16px", color: "rgba(255,255,255,0.5)" }}>
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Box>
        </Box>
      )}

      {/* ── No Alerts Message ── */}
      {crowdAlerts.length === 0 && (
        <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", textAlign: "center" }}>
          <CheckCircleIcon sx={{ fontSize: 48, color: "#34d399", mb: 2 }} />
          <Typography sx={{ color: "rgba(255,255,255,0.6)", fontSize: 14 }}>✅ No crowding incidents recorded</Typography>
        </Box>
      )}
    </Box>
  );
}
