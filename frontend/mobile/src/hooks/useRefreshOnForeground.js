import { useEffect } from "react";
import { AppState } from "react-native";

export function useRefreshOnForeground(refreshFn, enabled = true) {
  useEffect(() => {
    if (!enabled || !refreshFn) return undefined;

    const subscription = AppState.addEventListener("change", (nextState) => {
      if (nextState === "active") {
        refreshFn().catch(() => {});
      }
    });

    return () => subscription.remove();
  }, [refreshFn, enabled]);
}

