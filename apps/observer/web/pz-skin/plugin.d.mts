import type { Plugin } from 'vite';
export interface SkinConfig { skin: '3d' | 'theme' | 'off'; model: string; assetsDir: string }
export function skinConfig(env?: Record<string, string | undefined>): SkinConfig;
export function skinFiles(cfg?: SkinConfig): Array<[string, Uint8Array]>;
export function skinHtml(html: string, cfg?: SkinConfig): string;
export default function pzSkin(): Plugin;
