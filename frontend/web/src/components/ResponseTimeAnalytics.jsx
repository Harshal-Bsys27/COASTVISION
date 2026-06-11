import React, { useEffect, useState } from "react";
import { Box, Chip, LinearProgress, Stack, Typography } from "@mui/material";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import PersonIcon from "@mui/icons-material/Person";
import WavesIcon from "@mui/icons-material/Waves";

export default function ResponseTimeAnalytics({ api }) {
  const [rtData, setRtData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchResponseTimes = async () => {
      try {
        const res = await fetch(`${api}/api/analytics/response-times?limit=50`);
        const data = await res.json();
        setRtData(data);
      } catch (e) {
        console.error("Failed to fetch response times:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchResponseTimes();
    const interval = setInterval(fetchResponseTimes, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [api]);

  if (loading || !rtData) {
    return (
      <Box sx={{ py: 6, textAlign: "center" }}>
        <LinearProgress sx={{ bgcolor: "rgba(255,255,255,0.05)" }} />
        <Typography sx={{ color: "rgba(255,255,255,0.4)", fontSize: 13, mt: 2 }}>Loading response time data...</Typography>
      </Box>
    );
  }

  const { overall, by_zone, by_lifeguard, recent } = rtData;

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1fr 1fr 1fr" }, gap: 3 }}>
      
      {/* ── Overall Stats ── */}
      <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
          <AccessTimeIcon sx={{ fontSize: 28, color: "#2dd4bf" }} />
          <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>Overall Response Time</Typography>
        </Box>
        
        <Stack spacing={2}>
          <Box sx={{ display: "flex", justifyContent: "space-between", p: 2.5, borderRadius: 2, bgcolor: "rgba(45,212,191,0.08)", border: "1px solid rgba(45,212,191,0.15)" }}>
            <Typography sx={{ color: "rgba(255,255,255,0.6)", fontWeight: 600, fontSize: 14 }}>Average</Typography>
            <Typography sx={{ fontSize: 24, fontWeight: 900, color: "#2dd4bf" }}>{overall.avg_response_time}s</Typography>
          </Box>
          <Box sx={{ display: "flex", justifyContent: "space-between", p: 2.5, borderRadius: 2, bgcolor: "rgba(34,211,238,0.08)", border: "1px solid rgba(34,211,238,0.15)" }}>
            <Typography sx={{ color: "rgba(255,255,255,0.6)", fontWeight: 600, fontSize: 14 }}>Fastest</Typography>
            <Typography sx={{ fontSize: 24, fontWeight: 900, color: "#22d3ee" }}>{overall.min_response_time}s</Typography>
          </Box>
          <Box sx={{ display: "flex", justifyContent: "space-between", p: 2.5, borderRadius: 2, bgcolor: "rgba(244,114,182,0.08)", border: "1px solid rgba(244,114,182,0.15)" }}>
            <Typography sx={{ color: "rgba(255,255,255,0.6)", fontWeight: 600, fontSize: 14 }}>Slowest</Typography>
            <Typography sx={{ fontSize: 24, fontWeight: 900, color: "#f472b6" }}>{overall.max_response_time}s</Typography>
          </Box>
          <Box sx={{ display: "flex", justifyContent: "space-between", p: 2.5, borderRadius: 2, bgcolor: "rgba(52,211,153,0.08)", border: "1px solid rgba(52,211,153,0.15)" }}>
            <Typography sx={{ color: "rgba(255,255,255,0.6)", fontWeight: 600, fontSize: 14 }}>Total Responses</Typography>
            <Typography sx={{ fontSize: 24, fontWeight: 900, color: "#34d399" }}>{overall.total_responses}</Typography>
          </Box>
        </Stack>
      </Box>

      {/* ── By Lifeguard ── */}
      <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
          <PersonIcon sx={{ fontSize: 28, color: "#f59e0b" }} />
          <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>By Lifeguard</Typography>
        </Box>
        
        {Object.entries(by_lifeguard).length > 0 ? (
          <Stack spacing={1.5} sx={{ maxHeight: 400, overflowY: "auto" }}>
            {Object.entries(by_lifeguard).map(([lg_id, stats]) => (
              <Box key={lg_id} sx={{ p: 2, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
                <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#fff", mb: 1 }}>{stats.name}</Typography>
                <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, fontSize: 12 }}>
                  <Box><Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>Avg</Typography><Typography sx={{ color: "#2dd4bf", fontWeight: 700 }}>{stats.avg}s</Typography></Box>
                  <Box><Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>Responses</Typography><Typography sx={{ color: "#34d399", fontWeight: 700 }}>{stats.count}</Typography></Box>
                </Box>
              </Box>
            ))}
          </Stack>
        ) : (
          <Typography sx={{ color: "rgba(255,255,255,0.4)", textAlign: "center", py: 4, fontSize: 13 }}>No response data</Typography>
        )}
      </Box>

      {/* ── By Zone ── */}
      <Box sx={{ p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}>
          <WavesIcon sx={{ fontSize: 28, color: "#a78bfa" }} />
          <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff" }}>By Zone</Typography>
        </Box>
        
        {Object.entries(by_zone).length > 0 ? (
          <Stack spacing={1.5} sx={{ maxHeight: 400, overflowY: "auto" }}>
            {Object.entries(by_zone).map(([zone, stats]) => (
              <Box key={zone} sx={{ p: 2, borderRadius: 2, bgcolor: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
                <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#fff", mb: 1 }}>Zone {zone}</Typography>
                <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, fontSize: 12 }}>
                  <Box><Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>Avg</Typography><Typography sx={{ color: "#2dd4bf", fontWeight: 700 }}>{stats.avg}s</Typography></Box>
                  <Box><Typography sx={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>Alerts</Typography><Typography sx={{ color: "#34d399", fontWeight: 700 }}>{stats.count}</Typography></Box>
                </Box>
              </Box>
            ))}
          </Stack>
        ) : (
          <Typography sx={{ color: "rgba(255,255,255,0.4)", textAlign: "center", py: 4, fontSize: 13 }}>No zone data</Typography>
        )}
      </Box>

      {/* ── Recent Responses (Full Width) ── */}
      {recent.length > 0 && (
        <Box sx={{ gridColumn: "1 / -1", p: 4, borderRadius: 4, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)", boxShadow: "0 4px 20px rgba(0,0,0,0.25)" }}>
          <Typography sx={{ fontWeight: 800, fontSize: 22, color: "#fff", mb: 3 }}>Recent Responses</Typography>
          
          <Box sx={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Lifeguard</th>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Zone</th>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Response Time</th>
                  <th style={{ padding: "12px 16px", textAlign: "left", color: "rgba(255,255,255,0.6)", fontWeight: 700 }}>Time</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((r, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <td style={{ padding: "12px 16px", color: "#fff", fontWeight: 600 }}>{r.lifeguard_name}</td>
                    <td style={{ padding: "12px 16px", color: "rgba(255,255,255,0.8)" }}>Zone {r.zone}</td>
                    <td style={{ padding: "12px 16px" }}>
                      <Chip label={`${r.response_time_seconds}s`} size="small" sx={{ bgcolor: r.response_time_seconds < 30 ? "rgba(52,211,153,0.2)" : r.response_time_seconds < 60 ? "rgba(245,158,11,0.2)" : "rgba(244,114,182,0.2)", color: r.response_time_seconds < 30 ? "#34d399" : r.response_time_seconds < 60 ? "#f59e0b" : "#f472b6", fontWeight: 700, height: 24 }} />
                    </td>
                    <td style={{ padding: "12px 16px", color: "rgba(255,255,255,0.5)" }}>
                      {new Date(r.responded_at).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        </Box>
      )}
    </Box>
  );
}
