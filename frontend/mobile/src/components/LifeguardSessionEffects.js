import { useCallback } from "react";
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

  const handleStreamAlert = useCallback((alert) => {
    if (!isDrowningAlert(alert)) return;
    logInfo("SSE drowning alert — vibration", { zone: alert?.zone });
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy).catch(() => {});
  }, []);

  useLifeguardAlertStream(
    baseUrl,
    lifeguard?.id,
    sessionToken,
    handleStreamAlert,
    isAuthenticated
  );

  return null;
}
