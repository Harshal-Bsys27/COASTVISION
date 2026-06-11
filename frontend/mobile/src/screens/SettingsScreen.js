import React, { useEffect, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { ActivityIndicator, Button, Snackbar, Text, TextInput } from "react-native-paper";
import ConnectionBanner from "../components/ConnectionBanner";
import { useApiContext } from "../context/ApiContext";
import { usePollApi } from "../hooks/usePollApi";
import { POLL_HEALTH_MS } from "../shared/constants";
import { colors, spacing } from "../theme";

function formatAssignedZones(zones) {
  if (!Array.isArray(zones) || zones.length === 0) {
    return "All zones";
  }
  return zones.map((z) => `Zone ${z}`).join(", ");
}

export default function SettingsScreen() {
  const {
    baseUrl,
    api,
    ready,
    connected,
    health,
    lifeguard,
    assignedZones,
    saveBaseUrl,
    testConnection,
    logout,
    refreshLifeguard,
    setConnected,
    setHealth,
    clearConnection,
  } = useApiContext();
  const [urlInput, setUrlInput] = useState(baseUrl);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState("");
  const [loggingOut, setLoggingOut] = useState(false);
  const [refreshingProfile, setRefreshingProfile] = useState(false);

  const healthPoll = usePollApi(
    () => api.health(),
    POLL_HEALTH_MS,
    Boolean(baseUrl),
    null,
    "health"
  );

  useEffect(() => {
    setUrlInput(baseUrl);
  }, [baseUrl]);

  useEffect(() => {
    if (healthPoll.data) {
      setConnected(true);
      setHealth(healthPoll.data);
    }
    if (healthPoll.error) {
      clearConnection();
    }
  }, [healthPoll.data, healthPoll.error, setConnected, setHealth, clearConnection]);

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveBaseUrl(urlInput);
      setMessage("Server URL saved");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleRefreshProfile = async () => {
    setRefreshingProfile(true);
    try {
      const me = await refreshLifeguard();
      setMessage(
        me ? `Profile updated — ${formatAssignedZones(me.zones || [])}` : "Profile refreshed"
      );
    } catch (err) {
      setMessage(err.message || "Could not refresh profile");
    } finally {
      setRefreshingProfile(false);
    }
  };

  const handleLogout = async () => {
    setLoggingOut(true);
    try {
      await logout();
    } finally {
      setLoggingOut(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    try {
      if (urlInput !== baseUrl) {
        await saveBaseUrl(urlInput);
      }
      const result = await testConnection();
      setMessage(`Connected — ${result.zones ?? 0} zones, GPU: ${result.gpu_name || result.device}`);
    } catch (err) {
      clearConnection();
      setMessage(err.message);
    } finally {
      setTesting(false);
    }
  };

  if (!ready) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator color={colors.primary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ConnectionBanner visible={Boolean(baseUrl) && !connected && !healthPoll.loading} />
      <ScrollView contentContainerStyle={styles.content}>
        {lifeguard ? (
          <View style={styles.profileCard}>
            <Text variant="headlineSmall" style={styles.heading}>
              Lifeguard Account
            </Text>
            <Text style={styles.meta}>Name: {lifeguard.name}</Text>
            <Text style={styles.meta}>Phone: {lifeguard.phone || "—"}</Text>
            <Text style={styles.meta}>Assigned: {formatAssignedZones(assignedZones)}</Text>
            <Button
              mode="contained-tonal"
              onPress={handleRefreshProfile}
              loading={refreshingProfile}
              style={styles.refreshButton}
            >
              Refresh Profile
            </Button>
            <Button
              mode="outlined"
              onPress={handleLogout}
              loading={loggingOut}
              textColor={colors.danger}
              style={styles.logoutButton}
            >
              Sign Out
            </Button>
          </View>
        ) : null}

        <Text variant="headlineSmall" style={styles.heading}>
          Server Connection
        </Text>
        <Text style={styles.help}>
          Enter your laptop IP on the same Wi-Fi, e.g. http://192.168.1.105:8000
        </Text>

        <TextInput
          label="Server URL"
          value={urlInput}
          onChangeText={setUrlInput}
          mode="outlined"
          autoCapitalize="none"
          autoCorrect={false}
          style={styles.input}
          textColor={colors.text}
          outlineColor={colors.border}
          activeOutlineColor={colors.primary}
        />

        <View style={styles.actions}>
          <Button mode="contained" onPress={handleSave} loading={saving} style={styles.button}>
            Save URL
          </Button>
          <Button mode="outlined" onPress={handleTest} loading={testing} style={styles.button}>
            Test Connection
          </Button>
        </View>

        <View style={styles.statusCard}>
          <View style={styles.statusRow}>
            <View style={[styles.dot, { backgroundColor: connected ? colors.success : colors.danger }]} />
            <Text style={styles.statusText}>{connected ? "Connected" : "Disconnected"}</Text>
          </View>
          {health ? (
            <>
              <Text style={styles.meta}>Device: {health.device}</Text>
              <Text style={styles.meta}>GPU: {health.gpu_name || "N/A"}</Text>
              <Text style={styles.meta}>Zones: {health.zones ?? "—"}</Text>
              <Text style={styles.meta}>FPS cap: {health.fps ?? "—"}</Text>
            </>
          ) : (
            <Text style={styles.meta}>Save a URL and test the connection to see backend health.</Text>
          )}
        </View>
      </ScrollView>

      <Snackbar visible={Boolean(message)} onDismiss={() => setMessage("")} duration={3000}>
        {message}
      </Snackbar>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  centered: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: colors.background,
  },
  content: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
  },
  heading: {
    color: colors.text,
    fontWeight: "800",
    marginBottom: spacing.sm,
  },
  help: {
    color: colors.textMuted,
    marginBottom: spacing.md,
    lineHeight: 20,
  },
  input: {
    backgroundColor: colors.surface,
    marginBottom: spacing.md,
  },
  actions: {
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  button: {
    borderRadius: 10,
  },
  profileCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: spacing.lg,
  },
  refreshButton: {
    marginTop: spacing.md,
    borderRadius: 10,
  },
  logoutButton: {
    marginTop: spacing.sm,
    borderColor: colors.danger,
    borderRadius: 10,
  },
  statusCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: spacing.sm,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: spacing.sm,
  },
  statusText: {
    color: colors.text,
    fontWeight: "700",
  },
  meta: {
    color: colors.textMuted,
    marginBottom: 4,
  },
});
