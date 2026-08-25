'use client';

import { useEffect, useRef, useCallback } from 'react';
import { WS_BASE_URL, API_BASE_URL } from '@/lib/constants';
import { useLidarStore } from '@/stores/useLidarStore';

export function useWebSocketStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const {
    setIsConnected,
    setFrameData,
    playbackState,
    targetFps,
    setPlaybackState,
  } = useLidarStore();

  // Initial HTTP Fetch so data is loaded instantly even before WebSocket handshake
  const fetchInitialFrame = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/processing/frame/0`);
      if (res.ok) {
        const data = await res.json();
        if (data && data.metadata && data.map) {
          setFrameData(data);
        }
      }
    } catch (err) {
      console.warn('Initial REST frame fetch failed, waiting for WebSocket:', err);
    }
  }, [setFrameData]);

  const connect = useCallback(() => {
    try {
      const wsUrl = `${WS_BASE_URL}/ws/stream`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Connected to LiDAR WebSocket Stream');
        setIsConnected(true);
        // Request play stream immediately
        ws.send(JSON.stringify({ action: 'play', payload: { fps: 10.0, mode: 'foveated' } }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data && data.metadata && data.map) {
            setFrameData(data);
          }
        } catch (err) {
          console.error('Error parsing WebSocket frame:', err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;
        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = setTimeout(connect, 1500);
      };

      ws.onerror = (err) => {
        setIsConnected(false);
        ws.close();
      };
    } catch (err) {
      setIsConnected(false);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = setTimeout(connect, 1500);
    }
  }, [setIsConnected, setFrameData]);

  useEffect(() => {
    fetchInitialFrame();
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect, fetchInitialFrame]);

  const sendAction = useCallback((action: string, payload: any = {}) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, payload }));
    }
  }, []);

  const play = useCallback(
    (fps: number = targetFps, mode: string = 'foveated') => {
      setPlaybackState('running');
      sendAction('play', { fps, mode });
    },
    [sendAction, setPlaybackState, targetFps]
  );

  const pause = useCallback(() => {
    setPlaybackState('paused');
    sendAction('pause');
  }, [sendAction, setPlaybackState]);

  const stop = useCallback(() => {
    setPlaybackState('idle');
    sendAction('stop');
  }, [sendAction, setPlaybackState]);

  const seek = useCallback(
    (frameId: number) => {
      sendAction('seek', { frame_id: frameId });
    },
    [sendAction]
  );

  const setFps = useCallback(
    (fps: number) => {
      sendAction('set_fps', { fps });
    },
    [sendAction]
  );

  return {
    play,
    pause,
    stop,
    seek,
    setFps,
  };
}
