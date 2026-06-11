import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Box, Button, Chip, Stack, TextField, Typography } from "@mui/material";
import FiberManualRecordIcon from "@mui/icons-material/FiberManualRecord";
import { createApi } from "../../../shared/api.js";

const darkTextFieldSx = {
  "& .MuiOutlinedInput-root": {
    color: "#fff",
    bgcolor: "rgba(0,0,0,0.2)",
    "& fieldset": { borderColor: "rgba(255,255,255,0.1)" },
    "&:hover fieldset": { borderColor: "rgba(45,212,191,0.3)" },
    "&.Mui-focused fieldset": { borderColor: "#2dd4bf" },
  },
  "& .MuiInputLabel-root": { color: "rgba(255,255,255,0.5)" },
};

export default function LifeguardAccountsPanel({ api: apiBase }) {
  const client = useMemo(() => createApi(apiBase), [apiBase]);
  const [lifeguards, setLifeguards] = useState([]);
  const [zones, setZones] = useState([]);
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [savingZoneFor, setSavingZoneFor] = useState(null);

  const loadData = useCallback(async () => {
    try {
      const [lgData, zonesData] = await Promise.all([client.listLifeguards(), client.zones()]);
      setLifeguards(lgData.lifeguards || []);
      setZones((zonesData.items || []).filter((z) => Number.isFinite(z.id)).sort((a, b) => a.id - b.id));
    } catch (e) {
      console.error("Failed to load lifeguard accounts:", e);
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleCreate = async () => {
    if (!name.trim()) {
      setMessage("Name is required");
      return;
    }
    if (!phone.trim()) {
      setMessage("Phone number is required");
      return;
    }
    setCreating(true);
    setMessage("");
    try {
      const data = await client.lifeguardRegister(name.trim(), phone.trim());
      setMessage(`✓ Created account for ${data.name || name.trim()}`);
      setName("");
      setPhone("");
      await loadData();
      setTimeout(() => setMessage(""), 3000);
    } catch (e) {
      setMessage(e.message || "Failed to create account");
    } finally {
      setCreating(false);
    }
  };

  const handleZoneToggle = async (lg, zoneId) => {
    const zoneNum = Number(zoneId);
    const current = Array.isArray(lg.zones) ? [...lg.zones] : [];
    const next = current.includes(zoneNum)
      ? current.filter((z) => z !== zoneNum)
      : [...current, zoneNum].sort((a, b) => a - b);

    setSavingZoneFor(lg.id);
    try {
      await client.assignLifeguardZones(lg.id, next);
      await loadData();
    } catch (e) {
      setMessage(e.message || "Failed to assign zones");
    } finally {
      setSavingZoneFor(null);
    }
  };

  const formatZones = (assigned) => {
    if (!assigned || assigned.length === 0) return "All zones";
    return assigned.map((z) => `Zone ${z}`).join(", ");
  };

  const formatLastSeen = (lg) => {
    if (lg.online) return "Online now";
    if (!lg.last_seen) return "Never";
    const ago = Math.max(0, Math.round(Date.now() / 1000 - lg.last_seen));
    if (ago < 60) return `${ago}s ago`;
    if (ago < 3600) return `${Math.round(ago / 60)}m ago`;
    return `${Math.round(ago / 3600)}h ago`;
  };

  if (loading && lifeguards.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: "center", color: "rgba(255,255,255,0.5)" }}>
        Loading lifeguard accounts...
      </Box>
    );
  }

  return (
    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", lg: "1fr 1.4fr" }, gap: 3 }}>
      <Box sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)" }}>
        <Typography sx={{ fontWeight: 700, mb: 2, color: "#2dd4bf" }}>Create Lifeguard Account</Typography>
        <Stack spacing={2}>
          <TextField
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            size="small"
            fullWidth
            sx={darkTextFieldSx}
          />
          <TextField
            label="Phone number"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
            size="small"
            fullWidth
            placeholder="9876543210"
            sx={darkTextFieldSx}
          />
          <Button
            variant="contained"
            onClick={handleCreate}
            disabled={creating}
            sx={{
              background: "linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%)",
              color: "#071520",
              fontWeight: 700,
              textTransform: "none",
            }}
          >
            {creating ? "Creating..." : "Create Account"}
          </Button>
          <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.45)", lineHeight: 1.6 }}>
            Lifeguards sign in on the mobile app with this phone number. They cannot self-register.
          </Typography>
        </Stack>
      </Box>

      <Box sx={{ p: 3, borderRadius: 3, bgcolor: "#0f1923", border: "1px solid rgba(255,255,255,0.06)" }}>
        <Typography sx={{ fontWeight: 700, mb: 2, color: "#2dd4bf" }}>Staff Accounts ({lifeguards.length})</Typography>
        <Stack spacing={1.5}>
          {lifeguards.length === 0 ? (
            <Typography sx={{ fontSize: 13, color: "rgba(255,255,255,0.45)" }}>No lifeguard accounts yet.</Typography>
          ) : (
            lifeguards.map((lg) => (
              <Box key={lg.id} sx={{ p: 2, borderRadius: 2, bgcolor: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.1)" }}>
                <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ md: "center" }}>
                  <Box sx={{ flex: 1 }}>
                    <Typography sx={{ fontSize: 14, fontWeight: 700 }}>{lg.name}</Typography>
                    <Typography sx={{ fontSize: 12, color: "rgba(255,255,255,0.55)", mt: 0.5 }}>
                      Phone: {lg.phone || "—"} · {formatZones(lg.zones)}
                    </Typography>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.75 }}>
                      <FiberManualRecordIcon sx={{ fontSize: 10, color: lg.online ? "#34d399" : "rgba(255,255,255,0.35)" }} />
                      <Typography sx={{ fontSize: 11, color: lg.online ? "#34d399" : "rgba(255,255,255,0.45)" }}>
                        {formatLastSeen(lg)}
                      </Typography>
                    </Box>
                  </Box>
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    {zones.map((zone) => {
                      const selected = (lg.zones || []).includes(zone.id);
                      return (
                        <Chip
                          key={`${lg.id}-${zone.id}`}
                          label={zone.name || `Zone ${zone.id}`}
                          size="small"
                          clickable
                          disabled={savingZoneFor === lg.id}
                          onClick={() => handleZoneToggle(lg, zone.id)}
                          sx={{
                            bgcolor: selected ? "rgba(45,212,191,0.2)" : "rgba(255,255,255,0.05)",
                            color: selected ? "#5eead4" : "rgba(255,255,255,0.65)",
                            border: selected ? "1px solid rgba(45,212,191,0.45)" : "1px solid rgba(255,255,255,0.08)",
                            fontWeight: 600,
                          }}
                        />
                      );
                    })}
                  </Stack>
                </Stack>
              </Box>
            ))
          )}
        </Stack>
        {message && (
          <Typography sx={{ fontSize: 12, color: message.includes("✓") ? "#34d399" : "#ef4444", mt: 2, p: 1, bgcolor: message.includes("✓") ? "rgba(52,211,153,0.1)" : "rgba(239,68,68,0.1)", borderRadius: 1 }}>
            {message}
          </Typography>
        )}
      </Box>
    </Box>
  );
}
