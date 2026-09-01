'use client';

import { useEffect, useRef, useCallback } from 'react';
import { WS_BASE_URL, API_BASE_URL } from '@/lib/constants';
import { useLidarStore } from '@/stores/useLidarStore';

export function useWebSocketStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const localTickerRef = useRef<NodeJS.Timeout | null>(null);

  const isConnected = useLidarStore((state) => state.isConnected);
  const setIsConnected = useLidarStore((state) => state.setIsConnected);
  const setConnectionState = useLidarStore((state) => state.setConnectionState);
  const setFrameData = useLidarStore((state) => state.setFrameData);
  const playbackState = useLidarStore((state) => state.playbackState);
  const targetFps = useLidarStore((state) => state.targetFps);
  const setPlaybackState = useLidarStore((state) => state.setPlaybackState);
  const stepFrame = useLidarStore((state) => state.stepFrame);
  const setCurrentFrameIdx = useLidarStore((state) => state.setCurrentFrameIdx);

  // Initial HTTP Fetch so data is loaded instantly
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
      console.warn('Initial REST frame fetch, fallback to client-side engine');
    }
  }, [setFrameData]);

  const connect = useCallback(() => {
    try {
      setConnectionState('connecting');
      const wsUrl = `${WS_BASE_URL}/ws/stream`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setConnectionState('connected');
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
        setConnectionState('simulated');
        wsRef.current = null;
        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = setTimeout(() => {
          setConnectionState('reconnecting');
          connect();
        }, 3000);
      };

      ws.onerror = () => {
        setIsConnected(false);
        setConnectionState('simulated');
      };
    } catch (err) {
      setIsConnected(false);
      setConnectionState('simulated');
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = setTimeout(() => {
        setConnectionState('reconnecting');
        connect();
      }, 3000);
    }
  }, [setIsConnected, setConnectionState, setFrameData]);

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

  // Local fallback ticker ensuring seamless real-time playback
  useEffect(() => {
    if (playbackState === 'running' && !isConnected) {
      const intervalMs = Math.max(30, 1000 / targetFps);
      localTickerRef.current = setInterval(() => {
        stepFrame(1);
      }, intervalMs);
    } else {
      if (localTickerRef.current) {
        clearInterval(localTickerRef.current);
        localTickerRef.current = null;
      }
    }
    return () => {
      if (localTickerRef.current) {
        clearInterval(localTickerRef.current);
      }
    };
  }, [playbackState, isConnected, targetFps, stepFrame]);

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
    setCurrentFrameIdx(0);
    sendAction('stop');
  }, [sendAction, setPlaybackState, setCurrentFrameIdx]);

  const seek = useCallback(
    (frameId: number) => {
      setCurrentFrameIdx(frameId);
      sendAction('seek', { frame_id: frameId });
    },
    [sendAction, setCurrentFrameIdx]
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
