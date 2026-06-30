import React, { useEffect, useState } from "react";
import { ScrollView, StyleSheet, View, Image, Modal, TouchableOpacity, useWindowDimensions } from "react-native";
import { ActivityIndicator, Button, Snackbar, Text, TextInput, SegmentedButtons } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useIsFocused } from "@react-navigation/native";
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
  const { width } = useWindowDimensions();
  const isWide = width >= 920;
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
    streamQuality,
    saveStreamQuality,
  } = useApiContext();
  const isFocused = useIsFocused();
  const [urlInput, setUrlInput] = useState(baseUrl);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState("");
  const [loggingOut, setLoggingOut] = useState(false);
  const [refreshingProfile, setRefreshingProfile] = useState(false);
  const [avatarPreviewVisible, setAvatarPreviewVisible] = useState(false);

  const healthPoll = usePollApi(
    () => api.health(),
    POLL_HEALTH_MS,
    Boolean(baseUrl),
    null,
    "health"
  );

  const handleStreamQualityChange = async (value) => {
    await saveStreamQuality(value);
    setMessage(`Stream quality set to ${value} FPS`);
  };

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

  useEffect(() => {
    if (!isFocused || !lifeguard) return;
    refreshLifeguard().catch(() => {});
  }, [isFocused, lifeguard, refreshLifeguard]);

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
      <ScrollView contentContainerStyle={[styles.content, isWide && styles.contentWide]}>
        {lifeguard ? (
          <>
            <View style={[styles.pageHeader, isWide && styles.sectionCardWide]}>
              <Text style={styles.pageTitle}>Lifeguard Control Panel</Text>
              <Text style={styles.pageSubtitle}>Manage your profile, server access, and stream performance.</Text>
            </View>
            <View style={[styles.profileCard, isWide && styles.sectionCardWide]}>
              <View style={styles.profileHeader}>
                {lifeguard.avatar || lifeguard.avatar_thumb ? (
                  <TouchableOpacity onPress={() => setAvatarPreviewVisible(true)} activeOpacity={0.8}>
                    <Image source={{ uri: lifeguard.avatar_thumb || lifeguard.avatar }} style={styles.avatar} />
                  </TouchableOpacity>
                ) : (
                  <View style={styles.avatarFallback}>
                    <MaterialCommunityIcons name="account" size={28} color={colors.primary} />
                  </View>
                )}
                <Text variant="headlineSmall" style={styles.profileHeading}>
                  Lifeguard Account
                </Text>
                <Text style={styles.profileSub}>CoastVision mobile profile</Text>
              </View>
              <Modal
                visible={avatarPreviewVisible}
                transparent
                animationType="fade"
                onRequestClose={() => setAvatarPreviewVisible(false)}
              >
                <TouchableOpacity
                  style={styles.avatarModalOverlay}
                  activeOpacity={1}
                  onPress={() => setAvatarPreviewVisible(false)}
                >
                  <View style={styles.avatarModalContent}>
                    <Image
                      source={{ uri: lifeguard.avatar_thumb || lifeguard.avatar }}
                      style={styles.avatarPreview}
                      resizeMode="contain"
                    />
                    <Text style={styles.avatarPreviewCaption}>Tap anywhere to close</Text>
                  </View>
                </TouchableOpacity>
              </Modal>
              <View style={styles.profileDetails}>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Name</Text>
                  <Text style={styles.infoValue}>{lifeguard.name}</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Phone</Text>
                  <Text style={styles.infoValue}>{lifeguard.phone || "—"}</Text>
                </View>
                <View style={styles.infoRow}>
                  <Text style={styles.infoLabel}>Assigned</Text>
                  <Text style={styles.infoValue}>{formatAssignedZones(assignedZones)}</Text>
                </View>
              </View>
              <View style={[styles.profileActions, isWide && styles.profileActionsWide]}>
                <Button
                  mode="contained-tonal"
                  onPress={handleRefreshProfile}
                  loading={refreshingProfile}
                  style={[styles.refreshButton, isWide && styles.profileActionButtonWeb]}
                  contentStyle={[styles.actionContent, isWide && styles.actionContentWide]}
                >
                  Refresh Profile
                </Button>
                <Button
                  mode="outlined"
                  onPress={handleLogout}
                  loading={loggingOut}
                  textColor={colors.danger}
                  style={[styles.logoutButton, isWide && styles.profileActionButtonWeb]}
                  contentStyle={[styles.actionContent, isWide && styles.actionContentWide]}
                >
                  Sign Out
                </Button>
              </View>
            </View>
          </>
        ) : null}

        <View style={[styles.statusCard, isWide && styles.sectionCardWide]}>
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

          <View style={[styles.actions, isWide && styles.actionsWide]}>
            <Button
              mode="contained"
              onPress={handleSave}
              loading={saving}
              style={[styles.button, isWide && styles.buttonWeb]}
              contentStyle={[styles.actionContent, isWide && styles.actionContentWide]}
            >
              Save URL
            </Button>
            <Button
              mode="outlined"
              onPress={handleTest}
              loading={testing}
              style={[styles.button, isWide && styles.buttonWeb]}
              contentStyle={[styles.actionContent, isWide && styles.actionContentWide]}
            >
              Test Connection
            </Button>
          </View>

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

        <View style={styles.sectionCard}>
          <Text style={styles.heading}>Stream Quality</Text>
          <Text style={styles.help}>Choose video stream polling frequency. Higher FPS gives smoother playback; lower FPS conserves bandwidth and battery.</Text>
          <SegmentedButtons
            value={streamQuality}
            onValueChange={handleStreamQualityChange}
            buttons={[
              { value: "3", label: "3 FPS (Low)" },
              { value: "5", label: "5 FPS (Balanced)" },
              { value: "10", label: "10 FPS (High)" },
            ]}
            style={styles.segmentButtons}
          />
          <Text style={[styles.meta, { marginTop: spacing.md }]}>
            Current: {streamQuality} frames per second
          </Text>
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
  contentWide: {
    paddingHorizontal: spacing.lg,
    alignItems: "center",
  },
  sectionCard: {
    backgroundColor: colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  pageHeader: {
    backgroundColor: colors.surface,
    padding: spacing.md,
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    elevation: 2,
  },
  pageTitle: {
    color: colors.primary,
    fontSize: 22,
    fontWeight: "900",
    letterSpacing: 0.3,
    marginBottom: spacing.xs,
  },
  pageSubtitle: {
    color: colors.textMuted,
    fontSize: 13,
    lineHeight: 20,
    fontWeight: "600",
  },
  sectionCardWide: {
    width: "100%",
    maxWidth: 980,
  },
  heading: {
    color: colors.text,
    fontWeight: "800",
    marginBottom: spacing.sm,
    fontSize: 22,
  },
  help: {
    color: colors.textMuted,
    marginBottom: spacing.md,
    lineHeight: 20,
  },
  segmentButtons: {
    marginBottom: spacing.md,
  },
  input: {
    backgroundColor: colors.surface,
    marginBottom: spacing.md,
  },
  actions: {
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  actionsWide: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "flex-start",
  },
  button: {
    borderRadius: 12,
  },
  buttonWeb: {
    width: 130,
  },
  actionContent: {
    height: 44,
  },
  actionContentWide: {
    height: 36,
  },
  profileCard: {
    backgroundColor: colors.surface,
    borderRadius: 16,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: spacing.lg,
    shadowColor: "#000",
    shadowOpacity: 0.18,
    shadowRadius: 10,
    shadowOffset: { width: 0, height: 5 },
    elevation: 3,
  },
  profileHeader: {
    flexDirection: "column",
    alignItems: "center",
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  profileHeading: {
    color: colors.text,
    fontWeight: "900",
    fontSize: 24,
    marginTop: spacing.xs,
  },
  profileSub: {
    color: colors.textMuted,
    fontSize: 13,
    marginTop: 1,
    fontWeight: "600",
  },
  profileDetails: {
    borderRadius: 14,
    backgroundColor: "rgba(6,19,31,0.5)",
    borderWidth: 1,
    borderColor: "rgba(53,214,195,0.22)",
    padding: spacing.md,
    marginTop: spacing.xs,
    marginBottom: spacing.md,
    gap: 10,
  },
  infoRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.md,
  },
  infoLabel: {
    color: colors.textMuted,
    fontSize: 13,
    fontWeight: "700",
    minWidth: 84,
    letterSpacing: 0.2,
  },
  infoValue: {
    color: colors.text,
    flex: 1,
    textAlign: "right",
    fontSize: 16,
    fontWeight: "800",
    lineHeight: 22,
  },
  profileActions: {
    gap: spacing.sm,
  },
  profileActionsWide: {
    flexDirection: "row",
    alignItems: "center",
  },
  profileActionButtonWeb: {
    width: 132,
  },
  avatar: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 3,
    borderColor: "rgba(53,214,195,0.65)",
    backgroundColor: "rgba(16,34,53,0.9)",
  },
  avatarFallback: {
    width: 140,
    height: 140,
    borderRadius: 70,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(53,214,195,0.14)",
    borderWidth: 3,
    borderColor: "rgba(53,214,195,0.45)",
  },
  avatarModalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.55)",
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.md,
  },
  avatarModalContent: {
    width: "100%",
    maxWidth: 420,
    backgroundColor: colors.surface,
    borderRadius: 20,
    padding: spacing.md,
    alignItems: "center",
  },
  avatarPreview: {
    width: "100%",
    aspectRatio: 1,
    borderRadius: 20,
    marginBottom: spacing.sm,
    backgroundColor: colors.surface,
  },
  avatarPreviewCaption: {
    color: colors.textMuted,
    fontSize: 13,
    marginTop: spacing.xs,
    textAlign: "center",
  },
  refreshButton: {
    borderRadius: 10,
  },
  logoutButton: {
    borderColor: colors.danger,
    borderRadius: 10,
  },
  statusCard: {
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    width: "100%",
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
