const PREFIX = "[CoastVision]";

export function logInfo(message, data) {
  if (!__DEV__) return;
  if (data !== undefined) {
    console.log(PREFIX, message, data);
  } else {
    console.log(PREFIX, message);
  }
}

export function logWarn(message, data) {
  if (!__DEV__) return;
  if (data !== undefined) {
    console.warn(PREFIX, message, data);
  } else {
    console.warn(PREFIX, message);
  }
}

/** Use warn (not error) for non-fatal poll failures to avoid LogBox red screen. */
export function logError(message, error) {
  if (!__DEV__) return;
  if (error !== undefined) {
    console.warn(PREFIX, message, error?.message || error);
  } else {
    console.warn(PREFIX, message);
  }
}

export function logPollFailure(debugName, error) {
  if (!__DEV__) return;

  const details = {
    poll: debugName,
    message: error?.message || String(error),
    url: error?.url,
    method: error?.method || "GET",
    status: error?.status,
    responseBody: error?.responseBody,
    stack: error?.stack,
  };

  console.warn(PREFIX, `${debugName} poll failed`, details);
}

