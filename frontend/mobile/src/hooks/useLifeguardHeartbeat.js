import { useEffect } from "react";

const HEARTBEAT_MS = 60_000;

export function useLifeguardHeartbeat(api, lifeguardId, enabled = true) {
  useEffect(() => {
    if (!enabled || !lifeguardId || !api) return undefined;

    const send = () => {
      api.lifeguardHeartbeat(lifeguardId).catch(() => {});
    };

    send();
    const timer = setInterval(send, HEARTBEAT_MS);
    return () => clearInterval(timer);
  }, [api, lifeguardId, enabled]);
}
