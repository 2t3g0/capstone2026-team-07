interface StatusPillProps {
  tone: 'good' | 'warn' | 'bad' | 'neutral';
  children: React.ReactNode;
}

export function StatusPill({ tone, children }: StatusPillProps) {
  return <span className={`status-pill status-${tone}`}>{children}</span>;
}
