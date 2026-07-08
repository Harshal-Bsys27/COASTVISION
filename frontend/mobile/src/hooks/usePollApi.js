import { useCallback, useEffect, useRef, useState } from "react";
import { AppState } from "react-native";
import { useIsFocused } from "@react-navigation/native";
import { logPollFailure } from "../utils/logger";

const FRIENDLY_MESSAGES = {
  "crowd-alerts": "Unable to fetch crowd alerts. Retrying...",
  "crowd-status": "Unable to fetch crowd status. Retrying...",
  "response-times": "Unable to fetch response times. Retrying...",
  alerts: "Unable to fetch alerts. Retrying...",
  zones: "Unable to fetch zones. Retrying...",
  health: "Unable to reach server. Retrying...",
};

function toFriendlyError(debugName, err) {
  const friendly = FRIENDLY_MESSAGES[debugName] || "Unable to fetch data. Retrying...";
  const wrapped = new Error(friendly);
  wrapped.originalError = err;
  wrapped.isPollError = true;
  return wrapped;
}

export function usePollApi(
  fetcher,
  intervalMs,
  enabled = true,
  initialData = null,
  debugName = "api",
  { keepPreviousDataOnError = true } = {}
) {
  const isFocused = useIsFocused();
  const [data, setData] = useState(initialData);
  const [loading, setLoading] = useState(Boolean(enabled));
  const [error, setError] = useState(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(async () => {
    try {
      const result = await fetcherRef.current();
      setData(result);
      setError(null);
      return result;
    } catch (err) {
      logPollFailure(debugName, err);
      const friendly = toFriendlyError(debugName, err);
      setError(friendly);
      if (!keepPreviousDataOnError) {
        setData(initialData);
      }
      return null;
    } finally {
      setLoading(false);
    }
  }, [debugName, initialData, keepPreviousDataOnError]);

  useEffect(() => {
    if (!enabled || !isFocused) return undefined;

    let active = true;
    let timer = null;
    const appState = { current: AppState.currentState };

    const run = async () => {
      if (!active || appState.current !== "active" || !isFocused) return;
      try {
        const result = await fetcherRef.current();
        if (active) {
          setData(result);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        if (active) {
          logPollFailure(debugName, err);
          setError(toFriendlyError(debugName, err));
          setLoading(false);
        }
      }
    };

    run();
    timer = setInterval(run, intervalMs);

    const sub = AppState.addEventListener("change", (nextState) => {
      appState.current = nextState;
      if (nextState === "active" && isFocused) {
        run();
      }
    });

    return () => {
      active = false;
      if (timer) clearInterval(timer);
      sub.remove();
    };
  }, [debugName, enabled, intervalMs, isFocused]);

  return { data, loading, error, refresh };
}

