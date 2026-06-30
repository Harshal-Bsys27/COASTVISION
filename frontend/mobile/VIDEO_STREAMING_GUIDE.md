# Mobile Video Streaming: Technical Implementation Guide

## Current Status

**Active**: 15fps frame polling for smooth video-like playback
**Available**: HLS streaming infrastructure on backend
**Next**: Native video player implementation

## What Lifeguards See

### Dashboard (Zone Cards)
- **Display**: Smooth video feed at 15fps (was 1fps before)
- **Latency**: ~100-200ms (network dependent)
- **Quality**: Full resolution JPEG frames
- **Experience**: Looks like video playback now!

### Zone Detail (Full View)
- **Display**: Responsive full-width video stream
- **Size**: Fills available screen space (portrait/landscape)
- **Aspect Ratio**: 16:9 (standard video format)
- **Experience**: Professional cinema-like viewing
- **Below Stream**: Live detections, statistics, object list

## Technical Architecture

### Frame Polling Flow
```
Backend: Generates JPEG frames from video
    ↓
Endpoint: /api/zones/{zid}/frame.jpg
    ↓
Mobile: Polls every ~67ms (15fps)
    ↓
React Native: Image component renders
    ↓
Lifeguard: Sees smooth video animation
```

### Bandwidth Profile
- **Resolution**: ~640x480 per frame
- **Frame Size**: ~30-80KB (JPEG compressed)
- **15fps**: ~400-1200 KB/s (very manageable on 4G/5G)
- **Comparison**: Video streaming typically uses similar or more

## Why Current Approach Works Well

1. **Expo Go Compatible** ✅
   - No native modules required
   - Works in browser and simulator
   - No build process needed

2. **Efficient** ✅
   - Network: Similar to HLS
   - CPU: Minimal (native Image component)
   - Memory: Only 1 frame in memory

3. **Responsive** ✅
   - Adapts to screen size
   - Works on phones, tablets, landscape
   - No layout locks

4. **Reliable** ✅
   - HTTP fallback (no UDP issues)
   - Works through proxies/firewalls
   - Error recovery built-in

## Future: Native Video Player Setup

### Why Upgrade to HLS?

| Aspect | Current (15fps) | HLS Video |
|--------|-----------------|-----------|
| Latency | 100-200ms | 2-4 seconds |
| Quality | JPEG | H.264/H.265 |
| Bandwidth | 400KB-1MB/s | 500KB-2MB/s |
| CPU | Low | Medium |
| Setup | Simple | Complex |

**Decision**: Upgrade when you need <2s latency or better compression.

### Step-by-Step HLS Implementation

**1. Install Video Package**
```bash
cd frontend/mobile
npm install expo-av
# or
npm install react-native-video react-native-video-cache
```

**2. Create HLS Player Component**
```javascript
// frontend/mobile/src/components/HLSVideoPlayer.js
import React, { useRef } from 'react';
import { View, StyleSheet } from 'react-native';
import { Video } from 'expo-av';
import { ActivityIndicator, Text } from 'react-native-paper';
import { colors } from '../theme';

export default function HLSVideoPlayer({ 
  hlsUrl, 
  height = 300, 
  aspectRatio = 16/9 
}) {
  const videoRef = useRef(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState(null);

  if (!hlsUrl) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.fallback}>No stream URL</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.fallback}>Stream unavailable</Text>
      </View>
    );
  }

  return (
    <View style={[styles.container, { height }]}>
      <Video
        ref={videoRef}
        source={{ uri: hlsUrl }}
        style={styles.video}
        resizeMode="contain"
        isLooping
        shouldPlay
        onLoad={() => setIsLoading(false)}
        onError={() => {
          setIsLoading(false);
          setError(true);
        }}
        onLoadStart={() => setIsLoading(true)}
        progressUpdateIntervalMillis={500}
      />
      {isLoading && (
        <View style={styles.loader}>
          <ActivityIndicator color={colors.primary} />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.surfaceAlt,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border,
  },
  video: {
    width: '100%',
    height: '100%',
  },
  loader: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  fallback: {
    color: colors.textMuted,
    textAlign: 'center',
  },
});
```

**3. Update ZoneDetailScreen**
```javascript
// Change import
import HLSVideoPlayer from "../components/HLSVideoPlayer";

// Replace ZoneVideoPlayer with HLSVideoPlayer
<HLSVideoPlayer
  hlsUrl={api.hlsUrl(zone.id)}
  height={videoHeight}
/>
```

**4. Test**
```bash
# Test in Expo Go
npm start

# Test on Android
npm run android

# Test on iOS
npm run ios
```

### Troubleshooting HLS Issues

**Problem**: "Cannot load HLS stream"
- Check `/api/hls/status` endpoint returns valid data
- Verify backend is actually encoding HLS segments
- Check zone ID is correct

**Problem**: "Manifest not found"
- Ensure `/api/zones/{zid}/hls/stream.m3u8` exists
- Check backend HLS directory has write permissions
- Verify zone video file exists

**Problem**: "Playback stuck"
- Check network connectivity
- Verify FFmpeg installed on server (for encoding)
- Check server logs for encoding errors

## Switching Between Frame Polling and HLS

You can support both by detecting stream availability:

```javascript
const [useHLS, setUseHLS] = React.useState(false);

// Try HLS first
React.useEffect(() => {
  const checkHLS = async () => {
    try {
      const response = await fetch(api.hlsUrl(zone.id), { method: 'HEAD' });
      if (response.ok) {
        setUseHLS(true);
      }
    } catch {
      setUseHLS(false);
    }
  };
  checkHLS();
}, [zone.id, api]);

// Render
{useHLS ? (
  <HLSVideoPlayer hlsUrl={api.hlsUrl(zone.id)} />
) : (
  <ZoneVideoPlayer frameUrl={api.frameUrl(zone.id)} />
)}
```

## Performance Comparison

### 15fps Frame Polling (Current)
```
Network: GET /api/zones/1/frame.jpg + timestamp
Frequency: Every 67ms
Payload: ~50KB JPEG
Total BW: ~750KB/s @ 1920x1080
Latency: 100-200ms
CPU: ~2% (image decode)
Memory: ~5MB per stream
```

### HLS Video Streaming (Future)
```
Network: HTTP Live Streaming with .ts segments
Frequency: Every ~2-4s (segment based)
Payload: ~500KB per 3s segment (better compression)
Total BW: ~170-300KB/s @ 1920x1080
Latency: 2-4s (more stable)
CPU: ~8-15% (video decode)
Memory: ~15-30MB per stream
```

## Backend HLS Requirements

Your backend already supports HLS if:
1. ✅ `/api/hls/status` endpoint exists
2. ✅ Zone state has `hls_dir` property
3. ✅ FFmpeg is installed on server (for encoding)
4. ✅ Write permissions to HLS directory

Check status:
```bash
# From mobile
curl http://your-server:5000/api/hls/status
```

Expected response:
```json
{
  "hls_active": true,
  "zones": [
    {
      "id": 1,
      "hls_available": true,
      "segment_count": 5
    }
  ]
}
```

## Recommendation Timeline

**Now** → Keep 15fps polling
- Simple, effective, works everywhere
- Low setup, high compatibility
- Meets current requirements

**When** → Switch to HLS
- You need sub-2s latency
- Multiple zones are straining bandwidth
- Battery life on mobile becomes critical
- You want audio support

**After** → Consider advanced options
- Custom MJPEG player (middle ground)
- WebRTC streaming (real-time)
- GPU-accelerated decoding

## Migration Checklist

If upgrading to native video player:

- [ ] Install expo-av or react-native-video
- [ ] Create HLSVideoPlayer component
- [ ] Test on Android device
- [ ] Test on iOS device
- [ ] Test on web (if using react-native-video)
- [ ] Update ZoneDetailScreen import
- [ ] Test fallback to frame polling
- [ ] Update documentation
- [ ] Deploy to production
- [ ] Monitor performance/battery usage

## Related Code Locations

- Frame polling: `frontend/mobile/src/components/ZoneStream.js` (line 16)
- Video player: `frontend/mobile/src/components/ZoneVideoPlayer.js`
- Detail screen: `frontend/mobile/src/screens/ZoneDetailScreen.js`
- API methods: `frontend/mobile/src/shared/api.js` (lines 121-125)
- Backend HLS: `backend/server.py` (lines 1695-1740)

## Questions?

See `MOBILE_STREAMING_IMPROVEMENTS.md` for more details about current implementation.
