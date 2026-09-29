import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Execute the actual observation-page functions with delayed fetch/decode.
const source = readFileSync(new URL('../../ros2_ws/src/jolgwa_ros/jolgwa_ros/scenario_observe_view.py', import.meta.url), 'utf8');
const script = source.slice(source.indexOf('let imageUrl;'), source.indexOf('\nsetInterval(', source.indexOf('let imageUrl;')));
function page() {
  let now = 0, index = 0;
  const elements: Record<string, any> = { preview: { src: 'old', style: { opacity: '1' } }, 'video-status': { textContent: 'old' } };
  const decode = vi.fn(async () => {});
  const fetch = vi.fn(async (_path: string): Promise<any> => ({ ok: true, headers: { get: () => '100' }, blob: async () => ({}) }));
  const revoke = vi.fn();
  const context = { performance: { now: () => now }, fetch, AbortSignal, Date, Number, Error,
    URL: { createObjectURL: () => 'blob:' + ++index, revokeObjectURL: revoke },
    Image: class { src = ''; decode = decode; }, el: (id: string) => elements[id], setTimeout: vi.fn(), videoDeadline: 0 };
  runInNewContext(script + '\nglobalThis.poll=pollVideo;globalThis.get=getImage;', context);
  return { ...context, context: context as any, elements, decode, revoke, advance: (n: number) => { now += n; } };
}
describe('observation image delivery', () => {
  it('swaps only after decoding and retains a valid frame on failure', async () => {
    const p = page(); let done!: () => void;
    p.decode.mockImplementationOnce(() => new Promise<void>(r => { done = r; }));
    const pending = p.context.poll(true); await vi.waitFor(() => expect(done).toBeDefined());
    expect(p.elements.preview.src).toBe('old'); done(); await pending;
    expect(p.elements.preview.src).toBe('blob:1');
    p.fetch.mockRejectedValueOnce(new Error('network')); await p.context.poll(true);
    expect(p.elements.preview.style.opacity).toBe('1');
    p.advance(901); p.fetch.mockRejectedValueOnce(new Error('network')); await p.context.poll(true);
    expect(p.elements.preview.style.opacity).toBe('.25');
  });
  it('raw video works while annotation fetch is stuck', async () => {
    const p = page(); p.fetch.mockImplementationOnce(() => new Promise(() => {}));
    void p.context.poll(true); await p.context.poll(false);
    expect(p.elements['video-status'].textContent).toBe('원본 영상 · 주석 대기');
    expect(p.elements.preview.style.opacity).toBe('1');
  });
  it('request and decode time consume the original age lease', async () => {
    const p = page(); p.decode.mockImplementationOnce(async () => { p.advance(900); });
    await expect(p.context.get('/preview.jpg')).rejects.toThrow('preview_expired');
    expect(p.revoke).toHaveBeenCalledWith('blob:1');
    expect(p.elements.preview.src).toBe('old');
  });
  it('raw fallback never replaces a fresh annotation', async () => {
    const p = page(); await p.context.poll(true); await p.context.poll(false);
    expect(p.elements.preview.src).toBe('blob:1');
    p.advance(901); await p.context.poll(false);
    expect(p.elements['video-status'].textContent).toBe('원본 영상 · 주석 대기');
  });
});
