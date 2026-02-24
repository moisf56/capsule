/**
 * MedGemma Clinical Theme — Design Tokens
 *
 * Clean light theme for healthcare professionals.
 * Blue primary (trust/precision), tiered severity colors,
 * WCAG AA compliant text contrast on light backgrounds.
 */

// ─── Colors ──────────────────────────────────────────
export const C = {
  // Background layers (lightest → slightly darker for elevation)
  bg: {
    base:     '#F8FAFC',  // main app background (slate-50)
    card:     '#FFFFFF',  // card surfaces (white)
    elevated: '#F1F5F9',  // inputs, elevated elements (slate-100)
    header:   '#FFFFFF',  // header + bottom bar
  },

  // Primary — Medical Blue (trust, precision, focus)
  primary: {
    default: '#2563EB',  // buttons, links, active states (blue-600)
    dark:    '#1D4ED8',  // pressed variant (blue-700)
    soft:    '#DBEAFE',  // subtle blue tint backgrounds (blue-100)
  },

  // Accent — Teal (radiology, imaging features)
  accent: {
    default: '#0D9488',  // teal-600
    dark:    '#0F766E',  // teal-700
    soft:    '#CCFBF1',  // teal-100
  },

  // Severity — Tiered alert system (clinical standard)
  critical: {
    default: '#DC2626',  // drug interactions, life-threatening (red-600)
    light:   '#FCA5A5',
    bg:      '#FEF2F2',  // red-50
    border:  '#FECACA',  // red-200
  },
  warning: {
    default: '#D97706',  // cautions, disclaimers (amber-600)
    light:   '#FCD34D',
    bg:      '#FFFBEB',  // amber-50
    border:  '#FDE68A',  // amber-200
  },
  success: {
    default: '#16A34A',  // safe, approved (green-600)
    light:   '#86EFAC',
    bg:      '#F0FDF4',  // green-50
    border:  '#BBF7D0',  // green-200
  },

  // Text hierarchy
  text: {
    primary:   '#0F172A',  // main content (slate-900)
    secondary: '#475569',  // descriptions, metadata (slate-600)
    muted:     '#94A3B8',  // hints, timestamps (slate-400)
    code:      '#1D4ED8',  // ICD-10 codes, FHIR IDs (blue-700)
    inverse:   '#FFFFFF',  // text on colored backgrounds
  },

  // Borders
  border: {
    default: '#E2E8F0',  // card borders (slate-200)
    subtle:  '#F1F5F9',  // very faint separators (slate-100)
    focus:   '#2563EB',  // focused input borders
  },

  // Extras
  purple: '#7C3AED',  // audio file feature accent

  // Feature card icon backgrounds (soft tints)
  iconBg: {
    blue:    '#DBEAFE',  // dictation / SOAP (blue-100)
    amber:   '#FEF3C7',  // labs (amber-100)
    teal:    '#CCFBF1',  // imaging / radiology (teal-100)
    purple:  '#EDE9FE',  // chat (violet-100)
  },
};

// ─── Typography ──────────────────────────────────────
export const FONT = {
  title:   28,  // app name
  h1:      22,  // screen titles
  h2:      16,  // section headers / card titles
  body:    15,  // content text
  bodySmall: 13,  // descriptions in cards
  caption: 12,  // labels, hints
  mono:    14,  // SOAP notes, code, IDs
  label:   11,  // uppercase section labels
};

// ─── Workflow Steps ──────────────────────────────────
export const STEP_LABELS = ['Transcript', 'SOAP', 'Review', 'Export'] as const;

// Map screen names to step indices (0-based)
export const SCREEN_TO_STEP: Record<string, number> = {
  transcript: 0,
  soap:       1,
  alerts:     2,
  export:     3,
};
