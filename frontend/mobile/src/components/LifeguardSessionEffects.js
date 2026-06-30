import { useCallback } from "react";
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

  useLifeguardHeartbeat(api, lifeguard?.id, isAuthenticated);
  useRefreshOnForeground(refreshLifeguard, isAuthenticated);

  const handleStreamEvent = useCallback(
    async (payload) => {
      if (!payload) return;

      if (payload.type === "alert") {
        const alert = payload.alert || payload;
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
    [refreshLifeguard]
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
