import { useCallback, useEffect, useRef, useState } from 'react';
import { parseInboundMessage, serializeOutbound, type InboundMessage, type OutboundMessage } from '../lib/protocol';

export type SocketState = 'connecting' | 'connected' | 'disconnected' | 'error';

interface OperatorSocket {
  state: SocketState;
  lastError: string;
  send: (message: OutboundMessage) => boolean;
  reconnect: () => void;
}

export function useOperatorSocket(
  url: string,
  onMessage: (message: InboundMessage) => void,
): OperatorSocket {
  const [state, setState] = useState<SocketState>('connecting');
  const [lastError, setLastError] = useState('');
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const reconnectKeyRef = useRef(0);
  const [reconnectKey, setReconnectKey] = useState(0);
  const onMessageRef = useRef(onMessage);

  onMessageRef.current = onMessage;

  const reconnect = useCallback(() => {
    retryRef.current = 0;
    socketRef.current?.close();
    reconnectKeyRef.current += 1;
    setReconnectKey(reconnectKeyRef.current);
  }, []);

  useEffect(() => {
    let disposed = false;
    let retryTimer: number | undefined;

    const connect = () => {
      if (disposed) return;
      setState('connecting');
      let socket: WebSocket;
      try {
        socket = new WebSocket(url);
      } catch (error) {
        setState('error');
        setLastError(error instanceof Error ? error.message : 'WebSocket 생성 실패');
        return;
      }
      socketRef.current = socket;

      socket.onopen = () => {
        if (disposed || socketRef.current !== socket) return;
        retryRef.current = 0;
        setLastError('');
        setState('connected');
      };
      socket.onmessage = (event) => {
        if (disposed || socketRef.current !== socket) return;
        if (typeof event.data !== 'string') return;
        const message = parseInboundMessage(event.data);
        if (message) onMessageRef.current(message);
      };
      socket.onerror = () => {
        if (disposed || socketRef.current !== socket) return;
        setLastError('운영 게이트웨이에 연결할 수 없습니다.');
        setState('error');
      };
      socket.onclose = () => {
        if (disposed || socketRef.current !== socket) return;
        socketRef.current = null;
        setState('disconnected');
        const delay = Math.min(8_000, 500 * 2 ** retryRef.current++);
        retryTimer = window.setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      disposed = true;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [url, reconnectKey]);

  const send = useCallback((message: OutboundMessage) => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(serializeOutbound(message));
    return true;
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      send({ type: 'heartbeat', timestamp: Date.now() });
    }, 3_000);
    return () => window.clearInterval(timer);
  }, [send]);

  return { state, lastError, send, reconnect };
}
