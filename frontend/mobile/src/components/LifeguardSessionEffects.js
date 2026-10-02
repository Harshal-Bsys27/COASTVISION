import { useCallback, useEffect, useRef } from "react";
import { Alert } from "react-native";
import * as Haptics from "expo-haptics";
import { useApiContext } from "../context/ApiContext";
import { useLifeguardAlertStream } from "../hooks/useLifeguardAlertStream";
import { useLifeguardHeartbeat } from "../hooks/useLifeguardHeartbeat";
import { useRefreshOnForeground } from "../hooks/useRefreshOnForeground";
import { isDrowningAlert } from "../utils/format";
import { logInfo } from "../utils/logger";

export default function LifeguardSessionEffects() {
  const { baseUrl, api, sessionToken, lifeguard, isAuthenticated, refreshLifeguard } = useApiContext();
  const notifiedSosIds = useRef(new Set());

  const showSosNotification = useCallback((alert) => {
    const sosId = String(alert?.alert_id || alert?.event_id || "");
    if (!sosId || notifiedSosIds.current.has(sosId)) return;
    notifiedSosIds.current.add(sosId);
    Alert.alert(
      "🚨 EMERGENCY SOS",
      `${alert.created_by_name || "A lifeguard"} activated an emergency SOS in Zone ${alert.zone}. Respond immediately.`,
      [{ text: "Open Alert", onPress: () => {} }]
    );
  }, []);

  useEffect(() => {
    if (!isAuthenticated || !lifeguard?.id) return undefined;
    let active = true;
    api.lifeguardAlerts(lifeguard.id, 100)
      .then((data) => {
        if (!active) return;
        const sos = (data?.alerts || []).find(
          (alert) => String(alert?.category || "").toLowerCase() === "emergency_sos"
        );
        if (sos) showSosNotification(sos);
      })
      .catch(() => {});
    return () => { active = false; };
  }, [api, isAuthenticated, lifeguard?.id, showSosNotification]);

  useLifeguardHeartbeat(api, lifeguard?.id, isAuthenticated);
  useRefreshOnForeground(refreshLifeguard, isAuthenticated);

  const handleStreamEvent = useCallback(
    async (payload) => {
      if (!payload) return;

      if (payload.type === "alert") {
        const alert = payload.alert || payload;
        if (String(alert?.category || "").toLowerCase() === "emergency_sos") {
          showSosNotification(alert);
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
          return;
        }
        if (!isDrowningAlert(alert)) return;
        logInfo("SSE drowning alert — vibration", { zone: alert?.zone });
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy).catch(() => {});
        return;
      }

      if (payload.type === "assignment") {
        logInfo("SSE assignment update received", { zones: payload.zones });
        try {
          await refreshLifeguard();
          Alert.alert(
            "Assignment updated",
            payload.message || "Your assigned zones have changed.",
            [{ text: "OK" }]
          );
        } catch {
          // If refresh fails, still allow the app to continue.
        }
      }
    },
    [refreshLifeguard, showSosNotification]
  );

  useLifeguardAlertStream(
    baseUrl,
    lifeguard?.id,
    sessionToken,
    handleStreamEvent,
    isAuthenticated
  );

  return null;
}

