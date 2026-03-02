import React, {useState, useRef, useCallback, useEffect} from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  ScrollView,
  ActivityIndicator,
  Alert,
  PermissionsAndroid,
  Platform,
  Image,
  Animated,
  Easing,
  FlatList,
  KeyboardAvoidingView,
} from 'react-native';
import {initLlama, type LlamaContext} from 'llama.rn';
import {InferenceSession, Tensor} from 'onnxruntime-react-native';
import LiveAudioStream from 'react-native-live-audio-stream';
import RNFS from 'react-native-fs';
import {computeMelSpectrogram} from './src/MelSpectrogram';
import {ctcGreedyDecode, formatTranscription} from './src/CTCDecoder';
import {
  checkDrugInteractions,
  suggestCodes,
  exportFull,
  analyzeImage,
  exportDiagnosticReport,
  listObservations,
  seedDemoLabs,
  listPatients,
  seedDemoPatients,
  listDiagnosticReports,
  getDiagnosticReportImage,
  enhanceSoap,
  ehrNavigate,
  ehrNavigateStream,
  maskName,
  MCP_BASE,
  setMcpBase,
  testConnection,
  DEMO_SERVER_URL,
  DEFAULT_LOCAL_URL,
  type DDIResult,
  type EHRNavigateResult,
  type CodeSuggestion,
  type FullExportResult,
  type ImageAnalysisResult,
  type DiagnosticReportResult,
  type DiagnosticReportSummary,
  type LabObservation,
  type PatientSummary,
  type EnhanceSOAPResult,
} from './src/MCPClient';
import {launchImageLibrary} from 'react-native-image-picker';
import MEDASR_VOCAB from './src/medasr_vocab.json';
import {C, STEP_LABELS, SCREEN_TO_STEP} from './src/theme';

// ─── Constants ───────────────────────────────────────────
const MEDGEMMA_PATH = '/data/local/tmp/medgemma.gguf';
const MEDASR_PATH = '/data/local/tmp/medasr_int8.onnx';
const AUDIO_FILE_PATH = '/data/local/tmp/test_audio.wav';
const STOP_WORDS = [
  '<end_of_turn>',
  '<eos>',
  '</s>',
  '<|end|>',
  '<|eot_id|>',
  '\n\nNote:',
  '\n\nDisclaimer',
  '\n\nImportant',
  '\n\n---',
  '\n\nPlease note',
  '\n\nThis is',
  '\n\nRemember',
  '\n\nAdditional',
];
const VOCAB_SIZE = 512;

type Screen = 'home' | 'recording' | 'transcript' | 'soap' | 'alerts' | 'export'
  | 'radiology' | 'radiology_detail' | 'radiology_scans' | 'radiology_report' | 'radiology_export'
  | 'labs' | 'labs_detail'
  | 'chat' | 'settings';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;  // EHR Navigator agent reasoning (collapsible)
}

interface DictationRecord {
  id: string;
  timestamp: number;
  duration: string;
  transcript: string;
  soapGenerated: boolean;
  fhirExported: boolean;
}

// ─── Inline Components ───────────────────────────────────

/** Workflow step indicator bar */
function StepIndicator({currentStep}: {currentStep: number}) {
  return (
    <View style={styles.stepBar}>
      {STEP_LABELS.map((label, idx) => {
        const isActive = idx === currentStep;
        const isDone = idx < currentStep;
        return (
          <React.Fragment key={label}>
            {idx > 0 && (
              <View style={[styles.stepConnector, isDone && styles.stepConnectorDone]} />
            )}
            <View style={[
              styles.stepItem,
              isActive && styles.stepItemActive,
              isDone && styles.stepItemDone,
            ]}>
              <Text style={[
                styles.stepNum,
                isActive && styles.stepNumActive,
                isDone && styles.stepNumDone,
              ]}>
                {isDone ? '\u2713' : idx + 1}
              </Text>
              <Text style={[
                styles.stepLabel,
                isActive && styles.stepLabelActive,
                isDone && styles.stepLabelDone,
              ]}>
                {label}
              </Text>
            </View>
          </React.Fragment>
        );
      })}
    </View>
  );
}

/** Pulsing red dot for recording indicator */
function PulsingDot() {
  const opacity = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, {
          toValue: 0.25,
          duration: 600,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 1,
          duration: 600,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ]),
    );
    pulse.start();
    return () => pulse.stop();
  }, [opacity]);

  return (
    <View style={styles.pulseRow}>
      <Animated.View style={[styles.pulseDot, {opacity}]} />
      <Text style={styles.pulseText}>Recording</Text>
    </View>
  );
}

// ─── Utilities ───────────────────────────────────────────

function extractMedications(soapText: string): string[] {
  const drugPatterns = [
    /\b([a-z]+(?:in|ol|am|il|ne|te|ide|one|ine|ate|cin|lin|fen|pin|pam|lol|tan|min|tin|rin))\s*\d+\s*(?:mg|mcg|ml|units?)\b/gi,
    /\b([a-z]+(?:in|ol|am|il|ne|te|ide|one|ine|ate|cin|lin|fen|pin|pam|lol|tan|min|tin|rin))\s+(?:daily|twice|once|bid|tid|qid|prn)\b/gi,
  ];
  const found = new Set<string>();
  for (const pattern of drugPatterns) {
    let match;
    while ((match = pattern.exec(soapText)) !== null) {
      const drug = match[1].toLowerCase();
      if (drug.length >= 4 && !['within', 'begin', 'pain', 'main', 'certain', 'again', 'obtain', 'remain', 'contain', 'maintain', 'explain', 'brain', 'train', 'plain', 'strain', 'routine', 'examine', 'determine', 'decline', 'outline', 'combine', 'online', 'baseline', 'medicine'].includes(drug)) {
        found.add(drug);
      }
    }
  }
  return Array.from(found);
}

const SAMPLE_TRANSCRIPT =
  '58-year-old male presents with chest pain radiating to the left arm, ' +
  'onset 2 hours ago while climbing stairs. History of hypertension and ' +
  'type 2 diabetes. Currently on aspirin 81mg daily and metformin 1000mg ' +
  'twice daily. Started ibuprofen 400mg yesterday for back pain. ' +
  'Vitals: BP 158/92, HR 88, O2 sat 96% on room air. ' +
  'ECG shows ST depression in leads V4-V6. ' +
  'Troponin pending. Lungs clear bilaterally. ' +
  'Assessment: Acute coronary syndrome, rule out NSTEMI.';

function base64ToInt16(base64: string): Int16Array {
  const binaryStr = atob(base64);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

function int16ToFloat32(pcm: Int16Array): Float32Array {
  const f32 = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    f32[i] = pcm[i] / 32768.0;
  }
  return f32;
}

function stripThinkingTrace(text: string): string {
  let cleaned = text;

  // Strip <think>...</think> blocks
  cleaned = cleaned.replace(/<think>[\s\S]*?<\/think>/g, '');

  // Strip <unused94>thought ... model_output\n pattern (MedGemma 1.5 thinking)
  // The model outputs: <unused94>thought\n...thinking...\nmodel_output\n...actual answer...
  // We want everything after the last "model_output\n"
  const modelOutputIdx = cleaned.lastIndexOf('model_output\n');
  if (modelOutputIdx >= 0) {
    cleaned = cleaned.substring(modelOutputIdx + 'model_output\n'.length);
  } else {
    // Fallback: strip <unused*>thought block if "model_output" hasn't appeared yet
    // Match <unused>, <unused94>, <unused0>, etc.
    const unusedMatch = cleaned.match(/<unused\d*>/);
    if (unusedMatch) {
      // Still in thinking phase — return empty so UI shows spinner
      const afterUnused = cleaned.substring(unusedMatch.index || 0);
      if (!afterUnused.includes('model_output')) {
        return '';
      }
    }
  }

  // For SOAP notes: jump to SUBJECTIVE: if present
  const subjIdx = cleaned.indexOf('SUBJECTIVE:');
  if (subjIdx > 0) {
    cleaned = cleaned.substring(subjIdx);
  }
  cleaned = cleaned.replace(/^\s+/, '');

  // Trim disclaimers after PLAN: section
  const planIdx = cleaned.indexOf('PLAN:');
  if (planIdx >= 0) {
    const afterPlan = cleaned.substring(planIdx + 5);
    const disclaimerPatterns = [
      /\n\n(?:Note:|Disclaimer|Important|Please note|This is|Remember|Additional|I )/,
    ];
    for (const pattern of disclaimerPatterns) {
      const match = afterPlan.match(pattern);
      if (match && match.index !== undefined) {
        cleaned = cleaned.substring(0, planIdx + 5 + match.index);
      }
    }
  }
  return cleaned.trim();
}

// ─── Simple Markdown Renderer (bold, headers, bullets) ───
function renderMarkdown(text: string | null | undefined, baseStyle?: any): React.ReactElement {
  if (!text) return <Text style={baseStyle}>{''}</Text>;
  const lines = String(text).split('\n');
  const elements: React.ReactElement[] = [];
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    // Strip markdown headers (##, ###) → bold text
    const headerMatch = line.match(/^#{1,4}\s+(.+)/);
    if (headerMatch) {
      line = headerMatch[1];
      elements.push(
        <Text key={i} style={[baseStyle, {fontWeight: '700', fontSize: 15, marginTop: i > 0 ? 8 : 0}]}>
          {parseBold(line, baseStyle)}{'\n'}
        </Text>
      );
      continue;
    }
    // Bullet lines
    const bulletMatch = line.match(/^[-*]\s+(.+)/);
    if (bulletMatch) {
      elements.push(
        <Text key={i} style={baseStyle}>
          {'  \u2022 '}{parseBold(bulletMatch[1], baseStyle)}{'\n'}
        </Text>
      );
      continue;
    }
    // Regular line
    elements.push(
      <Text key={i} style={baseStyle}>
        {parseBold(line, baseStyle)}{'\n'}
      </Text>
    );
  }
  return <Text>{elements}</Text>;
}

function parseBold(text: string | null | undefined, baseStyle?: any): React.ReactNode[] {
  if (!text) return [''];
  const parts: React.ReactNode[] = [];
  const regex = /\*\*(.+?)\*\*/g;
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <Text key={match.index} style={[baseStyle, {fontWeight: '700'}]}>{match[1]}</Text>
    );
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts.length > 0 ? parts : [text];
}

// ─── Main App ────────────────────────────────────────────

export default function App() {
  // ─── State ───────────────────────────────────────────
  const [screen, setScreen] = useState<Screen>('home');
  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');

  // Recording
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState('00:00');
  const audioChunksRef = useRef<Int16Array[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const secondsRef = useRef(0);

  // MedASR
  const [medasrSession, setMedasrSession] = useState<InferenceSession | null>(null);
  const [transcript, setTranscript] = useState('');
  const [transcribing, setTranscribing] = useState(false);

  // MedGemma
  const [llamaContext, setLlamaContext] = useState<LlamaContext | null>(null);
  const [soapNote, setSoapNote] = useState('');
  const [generating, setGenerating] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [soapGenTime, setSoapGenTime] = useState<number | null>(null);

  // DDI Alerts
  const [ddiResult, setDdiResult] = useState<DDIResult | null>(null);
  const [checkingDDI, setCheckingDDI] = useState(false);
  const [extractedMeds, setExtractedMeds] = useState<string[]>([]);

  // ICD-10 Code Suggestions
  const [icdSuggestions, setIcdSuggestions] = useState<CodeSuggestion[]>([]);
  const [acceptedCodes, setAcceptedCodes] = useState<Set<string>>(new Set());
  const [loadingCodes, setLoadingCodes] = useState(false);

  // Agentic Enhancement (MedGemma + MCP Tools)
  const [enhanceResult, setEnhanceResult] = useState<EnhanceSOAPResult | null>(null);
  const [enhancing, setEnhancing] = useState(false);
  const [enhanceStep, setEnhanceStep] = useState('');

  // Server Settings
  const [serverMode, setServerMode] = useState<'local' | 'demo'>('local');
  const [localServerUrl, setLocalServerUrl] = useState(DEFAULT_LOCAL_URL);
  const [isConnected, setIsConnected] = useState<boolean | null>(null); // null = untested
  const [testingConn, setTestingConn] = useState(false);

  // FHIR Export
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<FullExportResult | null>(null);

  // Radiology (Vision)
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedImageUri, setSelectedImageUri] = useState<string | null>(null);
  const [imageType, setImageType] = useState<'chest_xray' | 'mri' | 'ct' | 'fundoscopy' | 'ecg' | 'dermatology' | 'pathology' | 'general'>('chest_xray');
  const [clinicalContext, setClinicalContext] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<ImageAnalysisResult | null>(null);
  const [radiologyExportResult, setRadiologyExportResult] = useState<DiagnosticReportResult | null>(null);
  const [exportingRadiology, setExportingRadiology] = useState(false);

  // Patients (fetched from FHIR, never persisted)
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [loadingPatients, setLoadingPatients] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [selectedPatientName, setSelectedPatientName] = useState('');

  // Lab Results
  const [labResults, setLabResults] = useState<LabObservation[]>([]);
  const [loadingLabs, setLoadingLabs] = useState(false);
  const [seedingLabs, setSeedingLabs] = useState(false);
  const [ehrNavigating, setEhrNavigating] = useState(false);
  const [ehrNavSession, setEhrNavSession] = useState(false);

  // Radiology scans (DiagnosticReports for selected patient)
  const [patientScans, setPatientScans] = useState<DiagnosticReportSummary[]>([]);
  const [loadingScans, setLoadingScans] = useState(false);

  // Dictation history (in-memory only)
  const [dictationHistory, setDictationHistory] = useState<DictationRecord[]>([]);

  // Chat
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatGenerating, setChatGenerating] = useState(false);
  const [chatStreamText, setChatStreamText] = useState('');
  const chatListRef = useRef<FlatList>(null);
  const [expandedReasoning, setExpandedReasoning] = useState<Set<number>>(new Set());
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [streamingStep, setStreamingStep] = useState('');
  const [chatRecording, setChatRecording] = useState(false);
  const [chatTranscribing, setChatTranscribing] = useState(false);

  // ─── Audio Recording ──────────────────────────────────
  const requestMicPermission = async (): Promise<boolean> => {
    if (Platform.OS !== 'android') return true;
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
      {
        title: 'Microphone Permission',
        message: 'MedGemma needs microphone access to record patient encounters.',
        buttonPositive: 'Allow',
      },
    );
    return granted === PermissionsAndroid.RESULTS.GRANTED;
  };

  const startRecording = useCallback(async () => {
    const hasPermission = await requestMicPermission();
    if (!hasPermission) {
      Alert.alert('Permission Denied', 'Microphone permission is required.');
      return;
    }
    audioChunksRef.current = [];
    secondsRef.current = 0;
    setRecordingTime('00:00');

    LiveAudioStream.init({
      sampleRate: 16000,
      channels: 1,
      bitsPerSample: 16,
      audioSource: 6,
      bufferSize: 4096,
    });

    LiveAudioStream.on('data', (base64Data: string) => {
      const pcm = base64ToInt16(base64Data);
      audioChunksRef.current.push(pcm);
    });

    LiveAudioStream.start();
    setIsRecording(true);

    timerRef.current = setInterval(() => {
      secondsRef.current += 1;
      const m = Math.floor(secondsRef.current / 60).toString().padStart(2, '0');
      const s = (secondsRef.current % 60).toString().padStart(2, '0');
      setRecordingTime(`${m}:${s}`);
    }, 1000);
  }, []);

  const stopRecording = useCallback(() => {
    LiveAudioStream.stop();
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // ─── MedASR (ONNX) ──────────────────────────────────
  const loadMedASR = useCallback(async () => {
    try {
      setLoadingStatus('Loading MedASR (101 MB)...');
      const session = await InferenceSession.create(`file://${MEDASR_PATH}`);
      setMedasrSession(session);
      return session;
    } catch (err: any) {
      Alert.alert('MedASR Load Error', `Make sure ONNX file exists at:\n${MEDASR_PATH}\n\nError: ${err.message}`);
      return null;
    }
  }, []);

  const runMedASRInference = useCallback(async (audioFloat: Float32Array): Promise<string> => {
    let session = medasrSession;
    if (!session) {
      session = await loadMedASR();
      if (!session) throw new Error('Failed to load MedASR');
    }
    setLoadingStatus('Computing mel spectrogram...');
    await new Promise(r => setTimeout(r, 50)); // Let UI render before heavy computation
    const mel = await computeMelSpectrogram(audioFloat, (pct) => {
      setLoadingStatus(`Computing mel spectrogram... ${pct}%`);
    });
    console.log(`Mel spectrogram: ${mel.timeSteps} frames x ${mel.nMels} mels`);
    setLoadingStatus('Running MedASR inference...');
    const inputTensor = new Tensor('float32', mel.data, [1, mel.timeSteps, mel.nMels]);
    const maskTensor = new Tensor('bool', Array.from({length: mel.timeSteps}, () => true), [1, mel.timeSteps]);
    const results = await session.run({input_features: inputTensor, attention_mask: maskTensor});
    const logitsKey = session.outputNames[0];
    const logitsTensor = results[logitsKey];
    const logitsData = logitsTensor.data as Float32Array;
    const outTimeSteps = logitsTensor.dims[1];
    console.log(`Logits: [1, ${outTimeSteps}, ${VOCAB_SIZE}]`);
    setLoadingStatus('Decoding transcript...');
    const rawText = ctcGreedyDecode(logitsData, outTimeSteps, VOCAB_SIZE, MEDASR_VOCAB as string[]);
    let formatted = rawText.replace(/▁/g, ' ').trim();
    formatted = formatTranscription(formatted);
    console.log('Transcript:', formatted);
    return formatted;
  }, [medasrSession, loadMedASR]);

  const transcribeAudio = useCallback(async () => {
    const chunks = audioChunksRef.current;
    if (chunks.length === 0) {
      Alert.alert('No Audio', 'No audio recorded. Please record first.');
      return;
    }
    setTranscribing(true);
    setLoadingStatus('Processing audio...');

    // Small delay to let UI show spinner before heavy computation
    await new Promise(r => setTimeout(r, 50));

    try {
      const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
      const allPcm = new Int16Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) { allPcm.set(chunk, offset); offset += chunk.length; }
      const audioFloat = int16ToFloat32(allPcm);
      console.log(`Audio: ${audioFloat.length} samples (${(audioFloat.length / 16000).toFixed(1)}s)`);
      const formatted = await runMedASRInference(audioFloat);
      setTranscript(formatted);
      setTranscribing(false);
      setLoadingStatus('');
      // Add to dictation history
      setDictationHistory(prev => [{
        id: Date.now().toString(),
        timestamp: Date.now(),
        duration: recordingTime,
        transcript: formatted,
        soapGenerated: false,
        fhirExported: false,
      }, ...prev]);
      setScreen('transcript');
      // Pre-load MedGemma while doctor reviews transcript
      if (!llamaContext) {
        console.log('Pre-loading MedGemma while doctor reviews transcript...');
        unloadMedASR();
        loadMedGemma(true).then(() => console.log('MedGemma pre-loaded'));
      }
    } catch (err: any) {
      console.error('Transcription error:', err);
      setTranscribing(false);
      setLoadingStatus('');
      Alert.alert('Transcription Error', err.message);
    }
  }, [runMedASRInference, recordingTime]);

  const transcribeFile = useCallback(async () => {
    setTranscribing(true);
    setLoadingStatus('Reading audio file...');
    try {
      const exists = await RNFS.exists(AUDIO_FILE_PATH);
      if (!exists) {
        throw new Error(`No audio file found at:\n${AUDIO_FILE_PATH}\n\nPush a WAV file with:\nadb push file.wav ${AUDIO_FILE_PATH}`);
      }
      const base64Data = await RNFS.readFile(AUDIO_FILE_PATH, 'base64');
      const binaryStr = atob(base64Data);
      const fileBytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) { fileBytes[i] = binaryStr.charCodeAt(i); }
      const view = new DataView(fileBytes.buffer);
      const numChannels = view.getUint16(22, true);
      const sampleRate = view.getUint32(24, true);
      const bitsPerSample = view.getUint16(34, true);
      console.log(`WAV: ${sampleRate}Hz, ${numChannels}ch, ${bitsPerSample}bit`);
      let dataOffset = 12;
      while (dataOffset < fileBytes.length - 8) {
        const chunkId = String.fromCharCode(fileBytes[dataOffset], fileBytes[dataOffset + 1], fileBytes[dataOffset + 2], fileBytes[dataOffset + 3]);
        const chunkSize = view.getUint32(dataOffset + 4, true);
        if (chunkId === 'data') { dataOffset += 8; break; }
        dataOffset += 8 + chunkSize;
      }
      const pcmBytes = fileBytes.slice(dataOffset);
      let audioFloat: Float32Array;
      if (bitsPerSample === 16) {
        const pcm16 = new Int16Array(pcmBytes.buffer, pcmBytes.byteOffset, Math.floor(pcmBytes.length / 2));
        audioFloat = int16ToFloat32(pcm16);
      } else if (bitsPerSample === 32) {
        audioFloat = new Float32Array(pcmBytes.buffer, pcmBytes.byteOffset, Math.floor(pcmBytes.length / 4));
      } else {
        throw new Error(`Unsupported bit depth: ${bitsPerSample}. Use 16-bit or 32-bit WAV.`);
      }
      if (numChannels === 2) {
        const mono = new Float32Array(Math.floor(audioFloat.length / 2));
        for (let i = 0; i < mono.length; i++) { mono[i] = audioFloat[i * 2]; }
        audioFloat = mono;
      }
      if (sampleRate !== 16000) {
        const ratio = sampleRate / 16000;
        const newLen = Math.floor(audioFloat.length / ratio);
        const resampled = new Float32Array(newLen);
        for (let i = 0; i < newLen; i++) { resampled[i] = audioFloat[Math.floor(i * ratio)]; }
        audioFloat = resampled;
      }
      console.log(`Audio file: ${audioFloat.length} samples (${(audioFloat.length / 16000).toFixed(1)}s)`);
      setLoadingStatus('Transcribing audio file...');
      const formatted = await runMedASRInference(audioFloat);
      setTranscript(formatted);
      setTranscribing(false);
      setLoadingStatus('');
      const duration = `${Math.floor(audioFloat.length / 16000 / 60).toString().padStart(2, '0')}:${Math.floor((audioFloat.length / 16000) % 60).toString().padStart(2, '0')}`;
      setDictationHistory(prev => [{
        id: Date.now().toString(),
        timestamp: Date.now(),
        duration,
        transcript: formatted,
        soapGenerated: false,
        fhirExported: false,
      }, ...prev]);
      setScreen('transcript');
      // Pre-load MedGemma while doctor reviews transcript
      if (!llamaContext) {
        console.log('Pre-loading MedGemma while doctor reviews transcript...');
        unloadMedASR();
        loadMedGemma(true).then(() => console.log('MedGemma pre-loaded'));
      }
    } catch (err: any) {
      console.error('File transcription error:', err);
      setTranscribing(false);
      setLoadingStatus('');
      Alert.alert('File Transcription Error', err.message);
    }
  }, [runMedASRInference]);

  const unloadMedASR = useCallback(() => {
    if (medasrSession) { medasrSession.release?.(); setMedasrSession(null); }
  }, [medasrSession]);

  // ─── MedGemma (llama.rn) ─────────────────────────────
  const loadMedGemma = useCallback(async (silent = false) => {
    try {
      if (!silent) setLoadingStatus('Loading MedGemma (2.3 GB)...\nThis takes 30-60 seconds');
      const ctx = await initLlama({
        model: `file://${MEDGEMMA_PATH}`,
        use_mlock: true,
        use_mmap: true,         // Memory-mapped loading (faster)
        n_ctx: 512,             // Halved context — saves KV cache memory, prompts are <300 tokens
        n_batch: 512,           // Larger batches for faster prompt prefill
        n_threads: 2,           // Pin to 2x big cores (Cortex-A75) — little cores add overhead
        n_gpu_layers: 0,        // CPU only (no GPU on this device)
      });
      setLlamaContext(ctx);
      return ctx;
    } catch (err: any) {
      Alert.alert('MedGemma Load Error', `Make sure GGUF file exists at:\n${MEDGEMMA_PATH}\n\nError: ${err.message}`);
      return null;
    }
  }, []);

  const generateSOAP = useCallback(async () => {
    if (!transcript.trim()) {
      Alert.alert('No Transcript', 'Please record and transcribe audio first.');
      return;
    }
    setGenerating(true);
    setStreamingText('');
    setSoapGenTime(null);
    setLoading(true);
    setLoadingStatus('Switching to MedGemma...');
    const genStart = Date.now();
    try {
      unloadMedASR();
      let ctx = llamaContext;
      if (!ctx) {
        ctx = await loadMedGemma();
        if (!ctx) { setGenerating(false); setLoading(false); return; }
      }
      setLoadingStatus('Generating SOAP note...');
      const soapPrompt = `Generate SOAP note grounded in transcription:

${transcript}`;

      let fullText = '';
      await ctx.completion(
        {
          messages: [
            {role: 'system', content: 'Medical Scribe: Output SOAP format for doctors. No thinking traces.'},
            {role: 'user', content: soapPrompt},
          ],
          n_predict: 330,
          stop: STOP_WORDS,
          temperature: 0.3,
          top_k: 40,
          top_p: 0.9,
          repeat_penalty: 1.0,   // Disabled — saves per-token compute, stop words catch runaway
        },
        data => {
          fullText += data.token;
          setStreamingText(fullText);  // Show raw stream (including thinking)
        },
      );
      let finalText = stripThinkingTrace(fullText);  // Clean only at end
      setSoapNote(finalText.trim());
      setSoapGenTime(Math.round((Date.now() - genStart) / 1000));
      setGenerating(false);
      setStreamingText('');
      setLoading(false);
      setLoadingStatus('');
      // Mark latest dictation as SOAP generated
      setDictationHistory(prev => prev.length > 0
        ? [{ ...prev[0], soapGenerated: true }, ...prev.slice(1)]
        : prev,
      );
      setScreen('soap');
    } catch (err: any) {
      setGenerating(false);
      setStreamingText('');
      setLoading(false);
      setLoadingStatus('');
      Alert.alert('Generation Error', err.message);
    }
  }, [transcript, llamaContext, loadMedGemma, unloadMedASR]);

  // ─── Chat helpers ──────────────────────────────────────
  const resetChatState = useCallback(async () => {
    // Cancel any ongoing generation
    if (chatGenerating && llamaContext) {
      try { await llamaContext.stopCompletion(); } catch {}
    }
    setChatMessages([]);
    setChatInput('');
    setChatGenerating(false);
    setChatStreamText('');
    setEhrNavigating(false);
    setExpandedReasoning(new Set());
    setStreamingReasoning('');
    setStreamingStep('');
  }, [chatGenerating, llamaContext]);

  // ─── Chat with MedGemma ───────────────────────────────
  const sendChatMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;
    const userMsg: ChatMessage = {role: 'user', content: text.trim()};
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput('');

    // ── EHR Navigator session: route through workstation agent (streaming) ──
    if (ehrNavSession && selectedPatientId) {
      setEhrNavigating(true);
      setStreamingReasoning('');
      setStreamingStep('');
      try {
        const result = await ehrNavigateStream(text.trim(), selectedPatientId, (event) => {
          if (event.step === 'done') return;
          // Stream reasoning updates to the UI
          setStreamingStep(event.label || event.step);
          if (event.reasoning) {
            setStreamingReasoning(event.reasoning);
          }
        });
        const meta = `\n\n---\nResources: ${result.resources_consulted.join(', ')}\nFacts: ${result.facts_extracted} | ${(result.processing_time_ms / 1000).toFixed(1)}s`;
        setChatMessages(prev => [...prev, {
          role: 'assistant',
          content: result.answer + meta,
          reasoning: result.reasoning || undefined,
        }]);
      } catch (e: any) {
        const msg = e?.message || String(e) || 'Unknown error';
        setChatMessages(prev => [...prev, {role: 'assistant', content: `EHR Navigator error: ${msg}\n\nMake sure llama-server is running on workstation.`}]);
      } finally {
        setEhrNavigating(false);
        setStreamingReasoning('');
        setStreamingStep('');
      }
      return;
    }

    // ── Normal on-device MedGemma chat ──
    setChatGenerating(true);
    setChatStreamText('');

    try {
      let ctx = llamaContext;
      if (!ctx) {
        setLoadingStatus('Loading MedGemma for chat...');
        unloadMedASR();
        ctx = await loadMedGemma();
        if (!ctx) { setChatGenerating(false); setLoadingStatus(''); return; }
        setLoadingStatus('');
      }

      let systemContent = 'Clinical AI assistant. Be concise. No thinking traces. Respond directly.';

      if (labResults.length > 0) {
        const labSummary = labResults.map(l =>
          `${l.loinc_display}: ${l.value} ${l.unit} [${l.interpretation}]`,
        ).join('\n');
        systemContent += `\n\nPatient lab results:\n${labSummary}`;
      }
      if (soapNote) {
        systemContent += `\n\nCurrent SOAP note:\n${soapNote}`;
      }

      const historyMessages = [...chatMessages, userMsg].slice(-3);  // Shorter history for faster TTFT
      const messages = [
        {role: 'system' as const, content: systemContent},
        ...historyMessages.map(m => ({role: m.role as 'user' | 'assistant', content: m.content})),
      ];

      let fullText = '';
      await ctx.completion(
        {
          messages,
          n_predict: 300,         // Chat answers don't need 500 tokens
          stop: STOP_WORDS,
          temperature: 0.4,
          top_k: 40,
          top_p: 0.9,
          repeat_penalty: 1.0,   // Disabled — saves per-token compute
        },
        (data: {token: string}) => {
          fullText += data.token;
          setChatStreamText(fullText);  // Show raw stream (including thinking)
        },
      );

      const finalText = stripThinkingTrace(fullText).trim();  // Clean only at end
      setChatMessages(prev => [...prev, {role: 'assistant', content: finalText}]);
      setChatGenerating(false);
      setChatStreamText('');
    } catch (err: any) {
      setChatGenerating(false);
      setChatStreamText('');
      Alert.alert('Chat Error', err.message);
    }
  }, [chatMessages, llamaContext, loadMedGemma, unloadMedASR, labResults, soapNote, ehrNavSession, selectedPatientId]);

  // ─── Load Patient List ───────────────────────────────
  const loadPatientList = useCallback(async () => {
    setLoadingPatients(true);
    try {
      const pts = await listPatients();
      setPatients(pts);
    } catch {
      setPatients([]);
    } finally {
      setLoadingPatients(false);
    }
  }, []);

  // ─── Chat Voice Input (MedASR) ──────────────────────
  const toggleChatVoice = useCallback(async () => {
    if (chatRecording) {
      // Stop recording and transcribe
      LiveAudioStream.stop();
      setChatRecording(false);
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }

      const chunks = audioChunksRef.current;
      if (chunks.length === 0) { return; }

      setChatTranscribing(true);
      try {
        // Unload MedGemma if needed to free RAM for MedASR
        if (llamaContext) {
          await llamaContext.release();
          setLlamaContext(null);
        }
        const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
        const allPcm = new Int16Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) { allPcm.set(chunk, offset); offset += chunk.length; }
        const audioFloat = int16ToFloat32(allPcm);
        const text = await runMedASRInference(audioFloat);
        setChatInput(prev => prev ? prev + ' ' + text : text);
      } catch (err: any) {
        Alert.alert('Voice Error', err.message);
      } finally {
        setChatTranscribing(false);
        setLoadingStatus('');
      }
    } else {
      // Start recording
      const hasPermission = await requestMicPermission();
      if (!hasPermission) { Alert.alert('Permission Denied', 'Microphone access is required.'); return; }

      // Pre-load MedASR while recording starts (so it's ready when we stop)
      if (!medasrSession) {
        loadMedASR().then(() => console.log('MedASR pre-loaded for chat voice'));
      }

      audioChunksRef.current = [];
      secondsRef.current = 0;
      setRecordingTime('00:00');

      LiveAudioStream.init({ sampleRate: 16000, channels: 1, bitsPerSample: 16, audioSource: 6, bufferSize: 4096 });
      LiveAudioStream.on('data', (base64Data: string) => {
        audioChunksRef.current.push(base64ToInt16(base64Data));
      });
      LiveAudioStream.start();
      setChatRecording(true);

      timerRef.current = setInterval(() => {
        secondsRef.current += 1;
        const m = Math.floor(secondsRef.current / 60).toString().padStart(2, '0');
        const s = (secondsRef.current % 60).toString().padStart(2, '0');
        setRecordingTime(`${m}:${s}`);
      }, 1000);
    }
  }, [chatRecording, llamaContext, medasrSession, loadMedASR, runMedASRInference, requestMicPermission]);

  // ─── Screen: Settings ────────────────────────────────
  if (screen === 'settings') {
    const handleTestConnection = async () => {
      setTestingConn(true);
      setIsConnected(null);
      const url = serverMode === 'demo' ? DEMO_SERVER_URL : localServerUrl;
      setMcpBase(url);
      const ok = await testConnection();
      setIsConnected(ok);
      setTestingConn(false);
    };

    const handleSave = () => {
      const url = serverMode === 'demo' ? DEMO_SERVER_URL : localServerUrl;
      setMcpBase(url);
      setScreen('home');
    };

    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.base} />
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('home')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Server Settings</Text>
        </View>

        <ScrollView contentContainerStyle={{padding: 20, gap: 20}}>
          {/* Mode toggle */}
          <View style={{backgroundColor: C.bg.card, borderRadius: 12, padding: 16}}>
            <Text style={[styles.sectionLabel, {marginBottom: 12}]}>Server Mode</Text>
            <View style={{flexDirection: 'row', gap: 10}}>
              {(['local', 'demo'] as const).map(mode => (
                <TouchableOpacity
                  key={mode}
                  onPress={() => { setServerMode(mode); setIsConnected(null); }}
                  style={{
                    flex: 1, paddingVertical: 12, borderRadius: 8, alignItems: 'center',
                    backgroundColor: serverMode === mode ? C.primary.default : C.bg.elevated,
                    borderWidth: 1,
                    borderColor: serverMode === mode ? C.primary.default : C.border.default,
                  }}>
                  <Text style={{
                    fontWeight: '600', fontSize: 14,
                    color: serverMode === mode ? '#fff' : C.text.secondary,
                  }}>
                    {mode === 'local' ? '🏠 Local (Production)' : '☁️ Judge Preview'}
                  </Text>
                  <Text style={{
                    fontSize: 11, marginTop: 2,
                    color: serverMode === mode ? 'rgba(255,255,255,0.8)' : C.text.muted,
                  }}>
                    {mode === 'local' ? 'Your laptop · full privacy' : 'Hackathon demo only'}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* URL input (local mode only) */}
          {serverMode === 'local' && (
            <View style={{backgroundColor: C.bg.card, borderRadius: 12, padding: 16}}>
              <Text style={[styles.sectionLabel, {marginBottom: 8}]}>Server URL</Text>
              <TextInput
                style={{
                  borderWidth: 1, borderColor: C.border.default, borderRadius: 8,
                  padding: 12, fontSize: 14, color: C.text.primary,
                  backgroundColor: C.bg.elevated, fontFamily: 'monospace',
                }}
                value={localServerUrl}
                onChangeText={v => { setLocalServerUrl(v); setIsConnected(null); }}
                placeholder="http://192.168.x.x:8082"
                placeholderTextColor={C.text.muted}
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
              />
              <Text style={{fontSize: 11, color: C.text.muted, marginTop: 6}}>
                Find your IP: run `hostname -I` on the server machine
              </Text>
            </View>
          )}

          {/* Demo server info */}
          {serverMode === 'demo' && (
            <View style={{backgroundColor: C.iconBg.blue, borderRadius: 12, padding: 16}}>
              <Text style={{fontSize: 13, color: C.primary.dark, fontWeight: '600', marginBottom: 4}}>
                ☁️ Demo Server
              </Text>
              <Text style={{fontSize: 12, color: C.primary.default, fontFamily: 'monospace'}}>
                {DEMO_SERVER_URL}
              </Text>
              <Text style={{fontSize: 11, color: C.text.secondary, marginTop: 6}}>
                For hackathon judges only. In production, everything runs on your local devices — no cloud, no PHI upload.
              </Text>
            </View>
          )}

          {/* Test connection */}
          <TouchableOpacity
            onPress={handleTestConnection}
            disabled={testingConn}
            style={{
              backgroundColor: testingConn ? C.bg.elevated : C.primary.default,
              borderRadius: 10, paddingVertical: 14, alignItems: 'center',
            }}>
            {testingConn ? (
              <View style={{flexDirection: 'row', alignItems: 'center', gap: 8}}>
                <ActivityIndicator size="small" color={C.primary.default} />
                <Text style={{color: C.primary.default, fontWeight: '600'}}>Testing...</Text>
              </View>
            ) : (
              <Text style={{color: '#fff', fontWeight: '600', fontSize: 15}}>
                Test Connection
              </Text>
            )}
          </TouchableOpacity>

          {/* Connection result */}
          {isConnected !== null && (
            <View style={{
              backgroundColor: isConnected ? C.success.bg : C.critical.bg,
              borderRadius: 10, padding: 14, flexDirection: 'row', alignItems: 'center', gap: 10,
            }}>
              <Text style={{fontSize: 20}}>{isConnected ? '✅' : '❌'}</Text>
              <View style={{flex: 1}}>
                <Text style={{
                  fontWeight: '700', fontSize: 14,
                  color: isConnected ? C.success.default : C.critical.default,
                }}>
                  {isConnected ? 'Connected successfully' : 'Connection failed'}
                </Text>
                <Text style={{fontSize: 12, color: C.text.secondary, marginTop: 2}}>
                  {isConnected
                    ? 'MCP server is reachable and responding.'
                    : 'Check that the server is running and the URL is correct.'}
                </Text>
              </View>
            </View>
          )}

          {/* Save / Apply */}
          <TouchableOpacity
            onPress={handleSave}
            style={{
              backgroundColor: C.success.default,
              borderRadius: 10, paddingVertical: 14, alignItems: 'center',
            }}>
            <Text style={{color: '#fff', fontWeight: '700', fontSize: 15}}>
              Save & Return
            </Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    );
  }

  // ─── Screen: Home ────────────────────────────────────
  if (screen === 'home') {
    const hour = new Date().getHours();
    const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

    const features = [
      {icon: '\uD83C\uDFA4', title: 'Dictate SOAP', desc: 'Voice to clinical notes', bg: C.iconBg.blue, onPress: () => {
        setScreen('recording');
        // Pre-load MedASR in background while user prepares to record
        if (!medasrSession) {
          loadMedASR().then(() => console.log('MedASR pre-loaded'));
        }
        // Pre-load MedGemma early so SOAP generation is instant
        if (!llamaContext) {
          console.log('Pre-loading MedGemma from recording screen...');
          loadMedGemma(true).then(() => console.log('MedGemma pre-loaded from recording'));
        }
      }},
      {icon: '\uD83D\uDD2C', title: 'Lab Results', desc: 'View & analyze labs', bg: C.iconBg.amber, onPress: () => {
        loadPatientList(); setScreen('labs');
      }},
      {icon: '\uD83D\uDCF7', title: 'Radiology Center', desc: 'X-ray & pathology AI', bg: C.iconBg.teal, onPress: () => {
        loadPatientList(); setScreen('radiology');
      }},
    ];

    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.base} />

        <ScrollView contentContainerStyle={styles.homeContent} keyboardShouldPersistTaps="handled">
          {/* Header area */}
          <View style={styles.homeHeader}>
            <View style={{flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start'}}>
              <View>
                <Text style={styles.homeGreeting}>{greeting}, Doctor</Text>
                <Text style={styles.homeTagline}>What would you like to do?</Text>
              </View>
              <TouchableOpacity
                onPress={() => setScreen('settings')}
                style={{padding: 8}}>
                <View style={{flexDirection: 'row', alignItems: 'center', gap: 6}}>
                  <View style={{
                    width: 8, height: 8, borderRadius: 4,
                    backgroundColor: isConnected === true ? C.success.default : isConnected === false ? C.critical.default : C.text.muted,
                  }} />
                  <Text style={{fontSize: 20}}>⚙️</Text>
                </View>
              </TouchableOpacity>
            </View>
          </View>

          {transcribing ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={C.primary.default} />
              <Text style={styles.loadingText}>{loadingStatus}</Text>
            </View>
          ) : (
            <View style={styles.homeGrid}>
              {features.map(f => (
                <TouchableOpacity key={f.title} style={styles.homeCard} activeOpacity={0.7} onPress={f.onPress}>
                  <View style={[styles.homeCardIcon, {backgroundColor: f.bg}]}>
                    <Text style={styles.homeCardEmoji}>{f.icon}</Text>
                  </View>
                  <Text style={styles.homeCardTitle}>{f.title}</Text>
                  <Text style={styles.homeCardDesc}>{f.desc}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}

          <Text style={styles.homeModelInfo}>
            MedASR (101 MB) + MedGemma 1.5 (2.3 GB){'\n'}All inference runs on-device
          </Text>
        </ScrollView>

        {/* Bottom input bar — always visible */}
        <View style={styles.homeInputBar}>
          <TouchableOpacity
            style={styles.homeInputMic}
            onPress={() => { setEhrNavSession(false); resetChatState(); setLabResults([]); setScreen('chat'); setTimeout(() => toggleChatVoice(), 300); }}>
            <Text style={styles.homeInputMicIcon}>{'\uD83C\uDFA4'}</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.homeInputField}
            activeOpacity={0.7}
            onPress={() => { setEhrNavSession(false); resetChatState(); setLabResults([]); setScreen('chat'); }}>
            <Text style={styles.homeInputPlaceholder}>Ask MedGemma anything...</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.homeInputSend, !chatInput.trim() && {opacity: 0.4}]}
            onPress={() => { if (chatInput.trim()) { setEhrNavSession(false); resetChatState(); setLabResults([]); sendChatMessage(chatInput); setScreen('chat'); } }}>
            <Text style={styles.homeInputSendIcon}>{'\u2192'}</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Recording ───────────────────────────────
  if (screen === 'recording') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => { stopRecording(); setScreen('home'); }}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Dictate SOAP</Text>
        </View>

        <ScrollView contentContainerStyle={{flexGrow: 1, padding: 16}}>
          {/* Dictation History */}
          {dictationHistory.length > 0 && (
            <View style={{marginBottom: 20}}>
              <Text style={styles.sectionLabel}>RECENT DICTATIONS</Text>
              {dictationHistory.slice(0, 5).map(rec => (
                <TouchableOpacity
                  key={rec.id}
                  style={styles.dictationRow}
                  onPress={() => {
                    setTranscript(rec.transcript);
                    setScreen('transcript');
                  }}>
                  <View style={{flex: 1}}>
                    <Text style={styles.dictationTime}>
                      {new Date(rec.timestamp).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}
                      {'  '}{rec.duration}
                    </Text>
                    <Text style={styles.dictationPreview} numberOfLines={1}>
                      {rec.transcript || 'No transcript'}
                    </Text>
                  </View>
                  <View style={styles.dictationBadges}>
                    <View style={[styles.dictationBadge, rec.soapGenerated ? {backgroundColor: C.success.bg, borderColor: C.success.border} : {backgroundColor: C.bg.elevated, borderColor: C.border.default}]}>
                      <Text style={[styles.dictationBadgeText, rec.soapGenerated ? {color: C.success.default} : {color: C.text.muted}]}>SOAP</Text>
                    </View>
                    <View style={[styles.dictationBadge, rec.fhirExported ? {backgroundColor: C.success.bg, borderColor: C.success.border} : {backgroundColor: C.bg.elevated, borderColor: C.border.default}]}>
                      <Text style={[styles.dictationBadgeText, rec.fhirExported ? {color: C.success.default} : {color: C.text.muted}]}>FHIR</Text>
                    </View>
                  </View>
                </TouchableOpacity>
              ))}
            </View>
          )}

          {/* Recording controls */}
          <View style={{alignItems: 'center', paddingVertical: 24}}>
            <Text style={styles.timerText}>{recordingTime}</Text>

            {isRecording && <PulsingDot />}

            <View style={styles.buttonRow}>
              {!isRecording ? (
                <TouchableOpacity style={[styles.circleButton, styles.recordBtn]} onPress={startRecording}>
                  <Text style={styles.circleButtonText}>REC</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity style={[styles.circleButton, styles.stopBtn]} onPress={stopRecording}>
                  <View style={styles.stopSquare} />
                </TouchableOpacity>
              )}
            </View>

            {!isRecording && audioChunksRef.current.length > 0 && (
              <View style={{marginTop: 24, width: '100%', paddingHorizontal: 16}}>
                <View style={styles.audioDurationChip}>
                  <Text style={styles.audioDurationText}>
                    {(audioChunksRef.current.reduce((s, c) => s + c.length, 0) / 16000).toFixed(1)}s recorded
                  </Text>
                </View>
                {transcribing ? (
                  <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color={C.primary.default} />
                    <Text style={styles.loadingText}>{loadingStatus}</Text>
                  </View>
                ) : (
                  <TouchableOpacity style={styles.primaryButton} onPress={transcribeAudio}>
                    <Text style={styles.primaryButtonText}>Transcribe with MedASR</Text>
                  </TouchableOpacity>
                )}
              </View>
            )}

            <Text style={styles.hint}>
              Speak clearly into the microphone.{'\n'}Describe the patient encounter as you normally would.
            </Text>
          </View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  // ─── Screen: Transcript Review (Step 1) ────────────
  if (screen === 'transcript') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('home')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Review Transcript</Text>
          <Text style={styles.headerSubtitle}>Step 1 of 4 {'\u2014'} Edit if needed</Text>
        </View>
        <StepIndicator currentStep={SCREEN_TO_STEP.transcript} />

        <ScrollView style={styles.scrollContent}>
          <Text style={styles.sectionLabel}>TRANSCRIPT</Text>
          <TextInput
            style={styles.transcriptInput}
            value={transcript}
            onChangeText={setTranscript}
            multiline
            textAlignVertical="top"
            placeholder="Transcript will appear here..."
            placeholderTextColor={C.text.muted}
          />
        </ScrollView>

        <View style={styles.bottomBar}>
          {loading || generating ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={C.primary.default} />
              <Text style={styles.loadingText}>{loadingStatus}</Text>
            </View>
          ) : (
            <TouchableOpacity style={styles.primaryButton} onPress={generateSOAP}>
              <Text style={styles.primaryButtonText}>Generate SOAP Note {'\u2192'}</Text>
            </TouchableOpacity>
          )}

          {streamingText !== '' && (
            <View style={styles.streamingPreview}>
              <Text style={styles.streamingLabel}>Generating...</Text>
              <Text style={styles.streamingText} numberOfLines={5}>{streamingText}</Text>
            </View>
          )}
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: SOAP Review (Step 2) ──────────────────
  if (screen === 'soap') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('transcript')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>SOAP Note</Text>
          <Text style={styles.headerSubtitle}>Step 2 of 4 {'\u2014'} Review and approve</Text>
        </View>
        <StepIndicator currentStep={SCREEN_TO_STEP.soap} />

        <ScrollView style={styles.scrollContent}>
          <Text style={styles.sectionLabel}>GENERATED NOTE{soapGenTime !== null ? `  (${soapGenTime}s)` : ''}</Text>
          <TextInput
            style={styles.soapInput}
            value={soapNote}
            onChangeText={setSoapNote}
            multiline
            textAlignVertical="top"
          />
        </ScrollView>

        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={[styles.primaryButton, {marginBottom: 8}]}
            disabled={enhancing}
            onPress={async () => {
              setEnhancing(true);
              setCheckingDDI(true);
              setLoadingCodes(true);
              // Progressive loading steps (timed sequence while backend processes)
              const ENHANCE_STEPS = [
                'Step 1/5: Extracting medications...',
                'Step 2/5: Checking drug interactions...',
                'Step 3/5: Suggesting ICD-10 codes...',
                'Step 4/5: Correlating lab results...',
                'Step 5/5: Generating clinical summary...',
              ];
              let stepIdx = 0;
              setEnhanceStep(ENHANCE_STEPS[0]);
              const stepTimer = setInterval(() => {
                stepIdx = Math.min(stepIdx + 1, ENHANCE_STEPS.length - 1);
                setEnhanceStep(ENHANCE_STEPS[stepIdx]);
              }, 2500);
              try {
                // Agentic enhancement: MedGemma + MCP tools
                const result = await enhanceSoap(soapNote, selectedPatientId || undefined);
                setEnhanceResult(result);
                // Populate existing state from enhance result
                setExtractedMeds(result.medications);
                setDdiResult({
                  found: result.ddi_alerts.length > 0,
                  summary: result.ddi_alerts.length > 0
                    ? `${result.ddi_alerts.filter(a => a.severity === 'critical').length} critical, ${result.ddi_alerts.length} total interaction(s)`
                    : 'No interactions found.',
                  interactions: result.ddi_alerts.map(a => ({
                    drug1: a.drug1, drug2: a.drug2, interaction_type: a.interaction_type,
                    severity: a.severity,
                  })),
                });
                setIcdSuggestions(result.icd10_suggestions.map(s => ({
                  code: s.code, description: s.description, matched_term: s.matched_term,
                })));
                setAcceptedCodes(new Set(result.icd10_suggestions.map(s => s.code)));
              } catch (err: any) {
                // Fallback: use local regex extraction + direct tool calls
                const meds = extractMedications(soapNote);
                setExtractedMeds(meds);
                const ddiPromise = meds.length >= 2
                  ? checkDrugInteractions(meds).catch(() => ({
                      found: false as const, summary: 'DDI check unavailable', interactions: [],
                    }))
                  : Promise.resolve({
                      found: false as const, summary: `${meds.length} med(s) found`, interactions: [],
                    });
                const codePromise = suggestCodes(soapNote).catch(() => ({
                  suggestions: [] as CodeSuggestion[], terms_searched: 0,
                }));
                const [ddiRes, codeRes] = await Promise.all([ddiPromise, codePromise]);
                setDdiResult(ddiRes);
                setIcdSuggestions(codeRes.suggestions);
                setAcceptedCodes(new Set(codeRes.suggestions.map(s => s.code)));
              }
              clearInterval(stepTimer);
              setEnhanceStep('');
              setEnhancing(false);
              setCheckingDDI(false);
              setLoadingCodes(false);
              setScreen('alerts');
            }}>
            <Text style={styles.primaryButtonText}>
              {enhancing ? (enhanceStep || 'Enhancing...') : 'Enhance'}
            </Text>
          </TouchableOpacity>
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.secondaryButton, {flex: 1, marginRight: 8}]}
              disabled={exporting}
              onPress={async () => {
                setExporting(true);
                try {
                  const patientId = selectedPatientId || '';
                  const result = await exportFull(patientId, 'Clinical encounter', soapNote, extractMedications(soapNote), [], []);
                  setExportResult(result);
                  setDictationHistory(prev => prev.length > 0
                    ? [{ ...prev[0], fhirExported: true }, ...prev.slice(1)]
                    : prev,
                  );
                  setScreen('export');
                } catch (err: any) {
                  Alert.alert('Export Failed', `Could not reach MCP server.\n\n${err.message}`);
                } finally {
                  setExporting(false);
                }
              }}>
              <Text style={styles.secondaryButtonText}>
                {exporting ? 'Exporting...' : 'Export'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, {flex: 1, marginLeft: 8}]}
              onPress={generateSOAP}>
              <Text style={styles.secondaryButtonText}>Regenerate</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Alert Review (Step 3 — CRITICAL) ───────
  if (screen === 'alerts') {
    const hasAlerts = ddiResult?.found === true;
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('soap')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Safety Review</Text>
          <Text style={styles.headerSubtitle}>Step 3 of 4 {'\u2014'} Review AI findings before export</Text>
        </View>
        <StepIndicator currentStep={SCREEN_TO_STEP.alerts} />

        {/* ── Sticky DDI Alert Banner (non-scrollable) ── */}
        {hasAlerts ? (
          <View style={{backgroundColor: C.critical.default, paddingVertical: 12, paddingHorizontal: 16, alignItems: 'center'}}>
            <Text style={{color: '#fff', fontSize: 15, fontWeight: '700', letterSpacing: 0.3}}>
              {ddiResult!.interactions.filter(i => i.severity === 'critical').length > 0
                ? `${ddiResult!.interactions.filter(i => i.severity === 'critical').length} CRITICAL DRUG INTERACTION${ddiResult!.interactions.filter(i => i.severity === 'critical').length > 1 ? 'S' : ''} DETECTED`
                : `${ddiResult!.interactions.length} DRUG INTERACTION${ddiResult!.interactions.length > 1 ? 'S' : ''} DETECTED`}
            </Text>
            <Text style={{color: '#fff', fontSize: 12, opacity: 0.9, marginTop: 2}}>
              Physician review required before export
            </Text>
          </View>
        ) : (
          <View style={{backgroundColor: C.success.default, paddingVertical: 10, paddingHorizontal: 16, alignItems: 'center'}}>
            <Text style={{color: '#fff', fontSize: 14, fontWeight: '600'}}>
              No drug interactions detected
            </Text>
          </View>
        )}

        <ScrollView style={styles.scrollContent}>
          {/* ── Count Summary Row ── */}
          <View style={{flexDirection: 'row', justifyContent: 'space-around', marginBottom: 16}}>
            <View style={{alignItems: 'center', flex: 1}}>
              <View style={{backgroundColor: C.primary.soft, borderRadius: 10, paddingHorizontal: 14, paddingVertical: 6}}>
                <Text style={{fontSize: 18, fontWeight: '700', color: C.primary.default, textAlign: 'center'}}>{extractedMeds.length}</Text>
              </View>
              <Text style={{fontSize: 10, color: C.text.muted, marginTop: 4, fontWeight: '600'}}>MEDS</Text>
            </View>
            <View style={{alignItems: 'center', flex: 1}}>
              <View style={{backgroundColor: hasAlerts ? C.critical.bg : C.success.bg, borderRadius: 10, paddingHorizontal: 14, paddingVertical: 6}}>
                <Text style={{fontSize: 18, fontWeight: '700', color: hasAlerts ? C.critical.default : C.success.default, textAlign: 'center'}}>
                  {ddiResult?.interactions.length || 0}
                </Text>
              </View>
              <Text style={{fontSize: 10, color: C.text.muted, marginTop: 4, fontWeight: '600'}}>DDI</Text>
            </View>
            <View style={{alignItems: 'center', flex: 1}}>
              <View style={{backgroundColor: '#EDE9FE', borderRadius: 10, paddingHorizontal: 14, paddingVertical: 6}}>
                <Text style={{fontSize: 18, fontWeight: '700', color: '#7C3AED', textAlign: 'center'}}>{icdSuggestions.length}</Text>
              </View>
              <Text style={{fontSize: 10, color: C.text.muted, marginTop: 4, fontWeight: '600'}}>ICD-10</Text>
            </View>
            {enhanceResult?.lab_findings && enhanceResult.lab_findings.length > 0 && (
              <View style={{alignItems: 'center', flex: 1}}>
                <View style={{backgroundColor: C.warning.bg, borderRadius: 10, paddingHorizontal: 14, paddingVertical: 6}}>
                  <Text style={{fontSize: 18, fontWeight: '700', color: C.warning.default, textAlign: 'center'}}>
                    {enhanceResult.lab_findings.filter(l => l.interpretation !== 'N').length}
                  </Text>
                </View>
                <Text style={{fontSize: 10, color: C.text.muted, marginTop: 4, fontWeight: '600'}}>LABS</Text>
              </View>
            )}
          </View>

          {/* AI Clinical Summary (MedGemma + MCP) */}
          {enhanceResult && (
            <>
              <Text style={styles.sectionLabel}>AI CLINICAL SUMMARY</Text>
              <View style={[styles.card, {borderLeftWidth: 4, borderLeftColor: C.primary.default}]}>
                {renderMarkdown(enhanceResult.clinical_summary, {fontSize: 14, color: C.text.primary, lineHeight: 20})}
                <View style={{flexDirection: 'row', flexWrap: 'wrap', marginTop: 10, gap: 6}}>
                  {enhanceResult.tools_called.map(tool => (
                    <View key={tool} style={{backgroundColor: C.primary.default + '20', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 10}}>
                      <Text style={{fontSize: 11, color: C.primary.default, fontWeight: '600'}}>
                        {tool.replace(/_/g, ' ').toUpperCase()}
                      </Text>
                    </View>
                  ))}
                  <Text style={{fontSize: 11, color: C.text.muted, alignSelf: 'center'}}>
                    {(enhanceResult.processing_time_ms / 1000).toFixed(1)}s
                  </Text>
                </View>
                {enhanceResult.medgemma_available && (
                  <Text style={{fontSize: 10, color: C.text.muted, marginTop: 6}}>
                    Powered by MedGemma + MCP
                  </Text>
                )}
              </View>
            </>
          )}

          {/* Medications as pills */}
          <Text style={[styles.sectionLabel, enhanceResult ? {marginTop: 20} : {}]}>MEDICATIONS DETECTED</Text>
          <View style={styles.card}>
            {extractedMeds.length > 0 ? (
              <View style={styles.medPillRow}>
                {extractedMeds.map(m => (
                  <View key={m} style={styles.medPill}>
                    <Text style={styles.medPillText}>
                      {m.charAt(0).toUpperCase() + m.slice(1)}
                    </Text>
                  </View>
                ))}
              </View>
            ) : (
              <Text style={styles.mutedText}>No medications detected in SOAP note.</Text>
            )}
          </View>

          {/* DDI Results */}
          <Text style={[styles.sectionLabel, {marginTop: 20}]}>DRUG INTERACTION CHECK</Text>
          {hasAlerts ? (
            <>
              <View style={styles.dangerCard}>
                <Text style={styles.dangerTitle}>{ddiResult!.summary}</Text>
                {ddiResult!.interactions.map((interaction, idx) => {
                  const isCrit = interaction.severity === 'critical';
                  return (
                    <View key={idx} style={styles.interactionRow}>
                      <View style={{flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 4}}>
                        <View style={{
                          backgroundColor: isCrit ? C.critical.default : C.warning.default,
                          paddingHorizontal: 8, paddingVertical: 2, borderRadius: 4,
                        }}>
                          <Text style={{color: '#fff', fontSize: 10, fontWeight: '800', letterSpacing: 0.5}}>
                            {isCrit ? 'CRITICAL' : 'MODERATE'}
                          </Text>
                        </View>
                        <Text style={styles.interactionDrugs}>
                          {interaction.drug1} + {interaction.drug2}
                        </Text>
                      </View>
                      <Text style={styles.interactionType}>{interaction.interaction_type}</Text>
                    </View>
                  );
                })}
              </View>
              <Text style={styles.alertWarning}>
                You must acknowledge these alerts before exporting.
              </Text>
            </>
          ) : (
            <View style={styles.safeCard}>
              <Text style={styles.safeText}>
                {ddiResult?.summary || 'No interaction data available.'}
              </Text>
            </View>
          )}

          {/* ICD-10 Billing Code Suggestions */}
          <Text style={[styles.sectionLabel, {marginTop: 20}]}>BILLING CODES (ICD-10)</Text>
          {loadingCodes ? (
            <View style={styles.card}>
              <ActivityIndicator color={C.primary.default} />
              <Text style={[styles.mutedText, {marginTop: 8}]}>Loading code suggestions...</Text>
            </View>
          ) : icdSuggestions.length > 0 ? (
            <View style={styles.card}>
              {/* Select All / Deselect All */}
              <View style={{flexDirection: 'row', justifyContent: 'flex-end', marginBottom: 8, gap: 12}}>
                <TouchableOpacity onPress={() => setAcceptedCodes(new Set(icdSuggestions.map(s => s.code)))}>
                  <Text style={{fontSize: 12, color: C.primary.default, fontWeight: '600'}}>Select All</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => setAcceptedCodes(new Set())}>
                  <Text style={{fontSize: 12, color: C.text.muted, fontWeight: '600'}}>Deselect All</Text>
                </TouchableOpacity>
              </View>
              {icdSuggestions.map((suggestion, idx) => (
                <TouchableOpacity
                  key={idx}
                  style={[styles.codeRow, acceptedCodes.has(suggestion.code) && styles.codeRowAccepted]}
                  onPress={() => {
                    setAcceptedCodes(prev => {
                      const next = new Set(prev);
                      if (next.has(suggestion.code)) { next.delete(suggestion.code); }
                      else { next.add(suggestion.code); }
                      return next;
                    });
                  }}>
                  <View style={styles.codeInfo}>
                    <Text style={styles.codeText}>{suggestion.code}</Text>
                    <Text style={styles.codeDesc}>{suggestion.description}</Text>
                    {suggestion.matched_term ? (
                      <Text style={{fontSize: 11, color: C.text.muted, fontStyle: 'italic', marginTop: 2}}>
                        matched: &quot;{suggestion.matched_term}&quot;
                      </Text>
                    ) : null}
                  </View>
                  <Text style={[styles.codeToggle, acceptedCodes.has(suggestion.code) && {color: C.success.default}]}>
                    {acceptedCodes.has(suggestion.code) ? '\u2713' : '\u25CB'}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          ) : (
            <View style={styles.safeCard}>
              <Text style={styles.safeText}>No billing codes suggested.</Text>
            </View>
          )}

          {/* Lab Correlations (from agentic enhancement) */}
          {enhanceResult?.lab_findings && enhanceResult.lab_findings.length > 0 && (
            <>
              <Text style={[styles.sectionLabel, {marginTop: 20}]}>LAB CORRELATIONS</Text>
              <View style={styles.card}>
                {enhanceResult.lab_findings.map((lab, idx) => {
                  const isHigh = lab.interpretation === 'H' || lab.interpretation === 'HH';
                  const isLow = lab.interpretation === 'L' || lab.interpretation === 'LL';
                  const isCritical = lab.interpretation === 'HH' || lab.interpretation === 'LL';
                  const flagColor = isHigh ? C.critical.default : isLow ? '#e65100' : C.text.muted;
                  const flagBg = isHigh ? C.critical.default + '15' : isLow ? '#e6510015' : '#f5f5f5';
                  return (
                    <View key={idx} style={{
                      flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
                      paddingVertical: 8, borderBottomWidth: idx < enhanceResult.lab_findings!.length - 1 ? 1 : 0,
                      borderBottomColor: C.border.default,
                    }}>
                      <View style={{flex: 1}}>
                        <Text style={{fontSize: 13, fontWeight: '600', color: C.text.primary}}>{lab.test}</Text>
                        <Text style={{fontSize: 12, color: C.text.muted}}>
                          {lab.reference_low != null && lab.reference_high != null
                            ? `Ref: ${lab.reference_low}-${lab.reference_high} ${lab.unit}`
                            : ''}
                        </Text>
                      </View>
                      <View style={{alignItems: 'flex-end'}}>
                        <Text style={{fontSize: 14, fontWeight: '700', color: flagColor}}>
                          {lab.value} {lab.unit}
                        </Text>
                        <View style={{backgroundColor: flagBg, paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4, marginTop: 2}}>
                          <Text style={{fontSize: 10, fontWeight: '700', color: flagColor}}>
                            {isCritical ? 'CRITICAL ' : ''}{isHigh ? 'HIGH' : isLow ? 'LOW' : 'NORMAL'}
                          </Text>
                        </View>
                      </View>
                    </View>
                  );
                })}
              </View>
            </>
          )}
        </ScrollView>

        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={[styles.primaryButton, hasAlerts && {backgroundColor: C.critical.default}]}
            disabled={exporting}
            onPress={async () => {
              setExporting(true);
              try {
                const patientId = selectedPatientId || '';
                const codes = icdSuggestions
                  .filter(s => acceptedCodes.has(s.code))
                  .map(s => ({code: s.code, description: s.description}));
                const alerts = (ddiResult?.interactions || []).map(i => ({
                  drug1: i.drug1, drug2: i.drug2, interaction_type: i.interaction_type, acknowledged: true,
                }));
                const result = await exportFull(patientId, 'Clinical encounter', soapNote, extractedMeds, codes, alerts);
                setExportResult(result);
                // Mark latest dictation as FHIR exported
                setDictationHistory(prev => prev.length > 0
                  ? [{ ...prev[0], fhirExported: true }, ...prev.slice(1)]
                  : prev,
                );
                setScreen('export');
              } catch (err: any) {
                Alert.alert('Export Failed', `Could not reach MCP server.\n\n${err.message}\n\nMake sure server is running at:\n${MCP_BASE}`);
              } finally {
                setExporting(false);
              }
            }}>
            <Text style={styles.primaryButtonText}>
              {exporting ? 'Exporting...' : hasAlerts ? 'Acknowledge & Export' : 'Approve & Export'}
            </Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Export Status (Step 4) ──────────────────
  if (screen === 'export') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <View style={styles.headerRow}>
            <TouchableOpacity onPress={() => setScreen('alerts')}>
              <Text style={styles.backButton}>{'\u2039'} Back</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => {
              setTranscript(''); setSoapNote(''); setDdiResult(null);
              setExtractedMeds([]); setIcdSuggestions([]); setAcceptedCodes(new Set());
              setExportResult(null); setSelectedImage(null); setSelectedImageUri(null);
              setAnalysisResult(null); setRadiologyExportResult(null);
              setLabResults([]); setChatMessages([]); setChatInput(''); setChatStreamText('');
              setScreen('home');
            }}>
              <Text style={styles.homeButton}>{'\u2302'} Home</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.headerTitle}>Export Complete</Text>
          <Text style={styles.headerSubtitle}>FHIR Resources Created</Text>
        </View>
        <StepIndicator currentStep={4} />

        <ScrollView style={styles.scrollContent}>
          {exportResult && (
            <>
              <View style={styles.successBanner}>
                <Text style={styles.successIcon}>{'\u2713'}</Text>
                <Text style={styles.successText}>{exportResult.summary}</Text>
              </View>

              <View style={styles.resourceCard}>
                <Text style={styles.resourceType}>ENCOUNTER</Text>
                <Text style={styles.resourceId}>ID: {exportResult.encounter?.id}</Text>
              </View>

              <View style={styles.resourceCard}>
                <Text style={styles.resourceType}>DOCUMENT REFERENCE (SOAP)</Text>
                <Text style={styles.resourceId}>ID: {exportResult.document?.id}</Text>
              </View>

              {exportResult.medications.map((med, idx) => (
                <View key={`med-${idx}`} style={styles.resourceCard}>
                  <Text style={styles.resourceType}>MEDICATION REQUEST</Text>
                  <Text style={styles.resourceId}>ID: {med.id}</Text>
                  <Text style={styles.resourceDetail}>
                    {med.drug} {med.rxnorm_code ? `(RxNorm: ${med.rxnorm_code})` : '(no RxNorm)'}
                  </Text>
                </View>
              ))}

              {exportResult.conditions.map((cond, idx) => (
                <View key={`cond-${idx}`} style={styles.resourceCard}>
                  <Text style={styles.resourceType}>CONDITION (DIAGNOSIS)</Text>
                  <Text style={styles.resourceId}>ID: {cond.id}</Text>
                  <Text style={styles.resourceDetail}>
                    ICD-10: {cond.icd_code}{cond.snomed_code ? ` | SNOMED: ${cond.snomed_code}` : ''}
                  </Text>
                </View>
              ))}

              {exportResult.detected_issues.map((di, idx) => (
                <View key={`di-${idx}`} style={[styles.resourceCard, styles.resourceCardDanger]}>
                  <Text style={styles.resourceType}>DETECTED ISSUE (SAFETY ALERT)</Text>
                  <Text style={styles.resourceId}>ID: {di.id}</Text>
                  <Text style={styles.resourceDetail}>{di.detail} [{di.severity}]</Text>
                </View>
              ))}

              {exportResult.errors.length > 0 && (
                <View style={styles.errorCard}>
                  <Text style={styles.errorTitle}>Warnings ({exportResult.errors.length})</Text>
                  {exportResult.errors.map((err, idx) => (
                    <Text key={idx} style={styles.errorText}>{err}</Text>
                  ))}
                </View>
              )}

              <Text style={styles.dashboardLink}>Dashboard: {MCP_BASE}/dashboard</Text>
            </>
          )}
        </ScrollView>

        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={() => {
              setTranscript(''); setSoapNote(''); setDdiResult(null);
              setExtractedMeds([]); setIcdSuggestions([]); setAcceptedCodes(new Set());
              setExportResult(null); setSelectedImage(null); setSelectedImageUri(null);
              setAnalysisResult(null); setRadiologyExportResult(null);
              setLabResults([]); setChatMessages([]); setChatInput(''); setChatStreamText('');
              setScreen('home');
            }}>
            <Text style={styles.primaryButtonText}>New Encounter</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Radiology — Patient List ─────────────────
  if (screen === 'radiology') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('home')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Radiology Center</Text>
          <Text style={styles.headerSubtitle}>FHIR Patients</Text>
        </View>

        <ScrollView style={styles.scrollContent}>
          {loadingPatients ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={C.accent.default} />
              <Text style={styles.loadingText}>Loading patients...</Text>
            </View>
          ) : patients.length === 0 ? (
            <View style={{alignItems: 'center', paddingVertical: 40}}>
              <Text style={styles.mutedText}>No patients found in FHIR server.</Text>
              <TouchableOpacity
                style={[styles.primaryButton, {marginTop: 16, backgroundColor: C.accent.default}]}
                disabled={seedingLabs}
                onPress={async () => {
                  setSeedingLabs(true);
                  try {
                    await seedDemoPatients();
                    await loadPatientList();
                  } catch (err: any) {
                    Alert.alert('Seed Error', err.message);
                  } finally {
                    setSeedingLabs(false);
                  }
                }}>
                <Text style={styles.primaryButtonText}>
                  {seedingLabs ? 'Seeding...' : 'Seed Demo Patients'}
                </Text>
              </TouchableOpacity>
            </View>
          ) : (
            <>
              {patients.map(pt => (
                <View key={pt.id} style={styles.patientRow}>
                  <View style={styles.patientInfo}>
                    <Text style={styles.patientName}>{maskName(pt.name)}</Text>
                    <Text style={styles.patientMeta}>
                      {pt.gender}  {pt.birthDate ? '\u00B7  ' + pt.birthDate.substring(0, 4) : ''}
                    </Text>
                  </View>
                  <View style={styles.patientActions}>
                    <TouchableOpacity
                      style={[styles.patientActionBtn, {borderColor: C.accent.default}]}
                      onPress={async () => {
                        setSelectedPatientId(pt.id);
                        setSelectedPatientName(pt.name);
                        setLoadingScans(true);
                        setPatientScans([]);
                        setScreen('radiology_scans');
                        try {
                          setPatientScans(await listDiagnosticReports(pt.id));
                        } catch { setPatientScans([]); }
                        finally { setLoadingScans(false); }
                      }}>
                      <Text style={[styles.patientActionText, {color: C.accent.default}]}>View Scans</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.patientActionBtn, {backgroundColor: C.accent.soft, borderColor: C.accent.default}]}
                      onPress={() => {
                        setSelectedPatientId(pt.id);
                        setSelectedPatientName(pt.name);
                        setSelectedImage(null);
                        setSelectedImageUri(null);
                        setAnalysisResult(null);
                        setScreen('radiology_detail');
                      }}>
                      <Text style={[styles.patientActionText, {color: C.accent.default}]}>New Scan</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              ))}

            </>
          )}
        </ScrollView>
        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={[styles.primaryButton, {backgroundColor: C.accent.default}]}
            onPress={() => setScreen('radiology_detail')}>
            <Text style={styles.primaryButtonText}>New Scan Analysis</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Radiology Scans — DiagnosticReports ────
  if (screen === 'radiology_scans') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('radiology')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{maskName(selectedPatientName)}</Text>
          <Text style={styles.headerSubtitle}>Diagnostic Reports</Text>
        </View>

        <ScrollView style={styles.scrollContent}>
          {loadingScans ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={C.accent.default} />
              <Text style={styles.loadingText}>Loading scans...</Text>
            </View>
          ) : patientScans.length === 0 ? (
            <View style={{alignItems: 'center', paddingVertical: 40}}>
              <Text style={styles.mutedText}>No diagnostic reports found.</Text>
            </View>
          ) : (
            patientScans.map((scan, idx) => (
              <TouchableOpacity
                key={scan.id || String(idx)}
                style={[styles.card, {marginBottom: 10, borderLeftWidth: 3, borderLeftColor: C.accent.default}]}
                onPress={async () => {
                  setLoadingScans(true);
                  try {
                    const imgData = await getDiagnosticReportImage(scan.id);
                    setSelectedImage(imgData.image_base64);
                    setSelectedImageUri(`data:${imgData.content_type};base64,${imgData.image_base64}`);
                    setImageType(scan.image_type as any || 'general');
                    setClinicalContext(scan.conclusion);
                    setScreen('radiology_detail');
                  } catch (err: any) {
                    Alert.alert('Load Error', err.message);
                  } finally {
                    setLoadingScans(false);
                  }
                }}>
                <Text style={styles.resourceType}>{scan.image_type?.toUpperCase() || 'DIAGNOSTIC REPORT'}</Text>
                <Text style={[styles.resourceDetail, {marginTop: 6}]}>{scan.conclusion}</Text>
                <Text style={[styles.mutedText, {fontSize: 11, marginTop: 4}]}>{scan.date}</Text>
                <Text style={{color: C.accent.default, fontSize: 12, fontWeight: '600', marginTop: 6}}>Tap to analyze with AI</Text>
              </TouchableOpacity>
            ))
          )}
        </ScrollView>

        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={[styles.primaryButton, {backgroundColor: C.accent.default}]}
            onPress={() => setScreen('radiology_detail')}>
            <Text style={styles.primaryButtonText}>New Scan Analysis</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Radiology Detail — Image Picker + Analyze ─────
  if (screen === 'radiology_detail') {
    const IMAGE_TYPES: Array<{key: typeof imageType; label: string}> = [
      {key: 'chest_xray', label: 'Chest X-ray'},
      {key: 'mri', label: 'MRI'},
      {key: 'ct', label: 'CT Scan'},
      {key: 'fundoscopy', label: 'Fundoscopy'},
      {key: 'ecg', label: 'ECG'},
      {key: 'dermatology', label: 'Dermatology'},
      {key: 'pathology', label: 'Pathology'},
      {key: 'general', label: 'General'},
    ];

    const pickImage = () => {
      launchImageLibrary(
        {mediaType: 'photo', includeBase64: true, maxWidth: 1024, maxHeight: 1024, quality: 0.8},
        response => {
          if (response.didCancel || response.errorCode) return;
          const asset = response.assets?.[0];
          if (asset?.base64 && asset?.uri) {
            setSelectedImage(asset.base64);
            setSelectedImageUri(asset.uri);
          }
        },
      );
    };

    const runAnalysis = async () => {
      if (!selectedImage) {
        Alert.alert('No Image', 'Please select a medical image first.');
        return;
      }
      setAnalyzing(true);
      try {
        const result = await analyzeImage(selectedImage, imageType, clinicalContext);
        setAnalysisResult(result);
        setScreen('radiology_report');
      } catch (err: any) {
        Alert.alert('Analysis Error', err.message);
      } finally {
        setAnalyzing(false);
      }
    };

    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('radiology')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Image Analysis</Text>
          <Text style={styles.headerSubtitle}>MedGemma Vision (workstation)</Text>
        </View>

        <ScrollView style={styles.scrollContent}>
          <Text style={styles.sectionLabel}>SELECT IMAGE</Text>
          <TouchableOpacity style={styles.imagePicker} onPress={pickImage}>
            {selectedImageUri ? (
              <Image source={{uri: selectedImageUri}} style={styles.imagePreview} resizeMode="contain" />
            ) : (
              <View style={styles.imagePickerPlaceholder}>
                <Text style={styles.imagePickerIcon}>+</Text>
                <Text style={styles.imagePickerText}>Tap to select from gallery</Text>
              </View>
            )}
          </TouchableOpacity>

          <Text style={[styles.sectionLabel, {marginTop: 20}]}>IMAGE TYPE</Text>
          <View style={styles.typeRow}>
            {IMAGE_TYPES.map(t => (
              <TouchableOpacity
                key={t.key}
                style={[styles.typePill, imageType === t.key && styles.typePillActive]}
                onPress={() => setImageType(t.key)}>
                <Text style={[styles.typePillText, imageType === t.key && styles.typePillTextActive]}>
                  {t.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={[styles.sectionLabel, {marginTop: 20}]}>CLINICAL CONTEXT (optional)</Text>
          <TextInput
            style={styles.contextInput}
            value={clinicalContext}
            onChangeText={setClinicalContext}
            placeholder="e.g., 65M with persistent cough, smoker for 30 years"
            placeholderTextColor={C.text.muted}
            multiline
          />
        </ScrollView>

        <View style={styles.bottomBar}>
          {analyzing ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={C.accent.default} />
              <Text style={styles.loadingText}>
                Analyzing with MedGemma Vision...{'\n'}This may take 30-60 seconds
              </Text>
            </View>
          ) : (
            <TouchableOpacity
              style={[styles.primaryButton, {backgroundColor: C.accent.default}, !selectedImage && {opacity: 0.4}]}
              disabled={!selectedImage}
              onPress={runAnalysis}>
              <Text style={styles.primaryButtonText}>Analyze Image</Text>
            </TouchableOpacity>
          )}
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Radiology Report — Review Findings ────
  if (screen === 'radiology_report') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('radiology_detail')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Image Analysis Report</Text>
          <Text style={styles.headerSubtitle}>Review AI findings</Text>
        </View>

        <ScrollView style={styles.scrollContent}>
          {selectedImageUri && (
            <Image source={{uri: selectedImageUri}} style={styles.reportThumbnail} resizeMode="contain" />
          )}

          {analysisResult && (
            <>
              <Text style={styles.sectionLabel}>FINDINGS</Text>
              <View style={styles.card}>
                {renderMarkdown(analysisResult.findings, styles.reportText)}
              </View>

              <Text style={[styles.sectionLabel, {marginTop: 20}]}>IMPRESSION</Text>
              <View style={[styles.card, {borderLeftWidth: 3, borderLeftColor: C.accent.default}]}>
                {renderMarkdown(analysisResult.impression, [styles.reportText, {color: C.accent.default}])}
              </View>

              <View style={styles.disclaimerCard}>
                <Text style={styles.disclaimerText}>
                  FOR CLINICAL DECISION SUPPORT ONLY{'\n'}
                  AI-generated findings must be verified by a qualified radiologist.
                </Text>
              </View>
            </>
          )}
        </ScrollView>

        <View style={styles.bottomBar}>
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.secondaryButton, {flex: 1, marginRight: 8, backgroundColor: C.accent.default, borderColor: C.accent.default}]}
              disabled={exportingRadiology}
              onPress={async () => {
                if (!analysisResult) return;
                setExportingRadiology(true);
                try {
                  const result = await exportDiagnosticReport(selectedPatientId || '1000', analysisResult.impression, analysisResult.findings, analysisResult.image_type);
                  setRadiologyExportResult(result);
                  setScreen('radiology_export');
                } catch (err: any) {
                  Alert.alert('Export Failed', err.message);
                } finally {
                  setExportingRadiology(false);
                }
              }}>
              <Text style={[styles.secondaryButtonText, {color: '#fff'}]}>
                {exportingRadiology ? 'Exporting...' : 'Export'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.secondaryButton, {flex: 1, marginLeft: 8}]}
              onPress={() => setScreen('radiology_detail')}>
              <Text style={styles.secondaryButtonText}>Regenerate</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Radiology Export Confirmation ──────────
  if (screen === 'radiology_export') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <View style={styles.headerRow}>
            <TouchableOpacity onPress={() => setScreen('radiology_report')}>
              <Text style={styles.backButton}>{'\u2039'} Back</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => {
              setSelectedImage(null); setSelectedImageUri(null);
              setImageType('chest_xray'); setClinicalContext('');
              setAnalysisResult(null); setRadiologyExportResult(null);
              setScreen('home');
            }}>
              <Text style={styles.homeButton}>{'\u2302'} Home</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.headerTitle}>Report Exported</Text>
          <Text style={styles.headerSubtitle}>FHIR DiagnosticReport Created</Text>
        </View>

        <ScrollView style={styles.scrollContent}>
          {radiologyExportResult && (
            <>
              <View style={styles.successBanner}>
                <Text style={styles.successIcon}>{'\u2713'}</Text>
                <Text style={styles.successText}>DiagnosticReport saved to FHIR</Text>
              </View>

              <View style={[styles.resourceCard, {borderLeftColor: C.accent.default}]}>
                <Text style={styles.resourceType}>DIAGNOSTIC REPORT</Text>
                <Text style={styles.resourceId}>ID: {radiologyExportResult.id}</Text>
                <Text style={styles.resourceDetail}>Status: {radiologyExportResult.status}</Text>
                <Text style={[styles.resourceDetail, {marginTop: 8}]}>
                  {radiologyExportResult.conclusion}
                </Text>
              </View>

              <Text style={styles.dashboardLink}>Dashboard: {MCP_BASE}/dashboard</Text>
            </>
          )}
        </ScrollView>

        <View style={styles.bottomBar}>
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={() => {
              setSelectedImage(null); setSelectedImageUri(null);
              setImageType('chest_xray'); setClinicalContext('');
              setAnalysisResult(null); setRadiologyExportResult(null);
              setScreen('home');
            }}>
            <Text style={styles.primaryButtonText}>Back to Home</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ─── Screen: Labs — Patient List ─────────────────────
  if (screen === 'labs') {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('home')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Lab Results</Text>
          <Text style={styles.headerSubtitle}>FHIR Patients</Text>
        </View>

        <ScrollView style={styles.scrollContent}>
          {loadingPatients ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={C.warning.default} />
              <Text style={styles.loadingText}>Loading patients...</Text>
            </View>
          ) : patients.length === 0 ? (
            <View style={{alignItems: 'center', paddingVertical: 40}}>
              <Text style={styles.mutedText}>No patients found in FHIR server.</Text>
              <TouchableOpacity
                style={[styles.primaryButton, {marginTop: 16, backgroundColor: C.warning.default}]}
                disabled={seedingLabs}
                onPress={async () => {
                  setSeedingLabs(true);
                  try {
                    await seedDemoPatients();
                    await loadPatientList();
                  } catch (err: any) {
                    Alert.alert('Seed Error', err.message);
                  } finally {
                    setSeedingLabs(false);
                  }
                }}>
                <Text style={styles.primaryButtonText}>
                  {seedingLabs ? 'Seeding...' : 'Seed Demo Patients'}
                </Text>
              </TouchableOpacity>
            </View>
          ) : (
            patients.map(pt => (
              <View key={pt.id} style={[styles.patientRow, {flexDirection: 'column', alignItems: 'stretch'}]}>
                <View style={{flexDirection: 'row', alignItems: 'center', marginBottom: 10}}>
                  <Text style={styles.patientName}>{maskName(pt.name)}</Text>
                  <Text style={[styles.patientMeta, {marginTop: 0, marginLeft: 10}]}>
                    {pt.gender}  {pt.birthDate ? '\u00B7  ' + pt.birthDate.substring(0, 4) : ''}
                  </Text>
                </View>
                <View style={[styles.patientActions, {justifyContent: 'flex-start'}]}>
                  <TouchableOpacity
                    style={styles.patientActionBtn}
                    onPress={async () => {
                      setSelectedPatientId(pt.id);
                      setSelectedPatientName(pt.name);
                      setLoadingLabs(true);
                      setLabResults([]);
                      setScreen('labs_detail');
                      try {
                        setLabResults(await listObservations(pt.id));
                      } catch { setLabResults([]); }
                      finally { setLoadingLabs(false); }
                    }}>
                    <Text style={styles.patientActionText}>View Labs</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.patientActionBtn, {backgroundColor: C.primary.soft, borderColor: C.primary.default}]}
                    onPress={async () => {
                      setSelectedPatientId(pt.id);
                      setSelectedPatientName(pt.name);
                      setEhrNavSession(false);
                      await resetChatState();
                      setLoadingLabs(true);
                      try {
                        const obs = await listObservations(pt.id);
                        setLabResults(obs);
                        if (obs.length > 0) {
                          const abnormal = obs.filter(l => l.interpretation !== 'N');
                          const summary = abnormal.length > 0
                            ? abnormal.map(l => `${l.loinc_display}: ${l.value} ${l.unit} (${l.interpretation})`).join(', ')
                            : obs.map(l => `${l.loinc_display}: ${l.value} ${l.unit}`).join(', ');
                          sendChatMessage(`Review these lab results for patient ${maskName(pt.name)} and provide clinical interpretation:\n${summary}`);
                        } else {
                          sendChatMessage(`No lab results available for patient ${maskName(pt.name)}. What lab tests would you recommend?`);
                        }
                      } catch { /* ignore */ }
                      finally { setLoadingLabs(false); }
                      setScreen('chat');
                    }}>
                    <Text style={[styles.patientActionText, {color: C.primary.default}]}>AI Summary</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[styles.patientActionBtn, {backgroundColor: '#E8F5E9', borderColor: '#2E7D32'}]}
                    onPress={async () => {
                      setSelectedPatientId(pt.id);
                      setSelectedPatientName(pt.name);
                      await resetChatState();
                      setEhrNavSession(true);
                      setLabResults([]);
                      setScreen('chat');
                    }}>
                    <Text style={[styles.patientActionText, {color: '#2E7D32'}]}>EHR Nav</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ))
          )}
        </ScrollView>
      </SafeAreaView>
    );
  }

  // ─── Screen: Labs Detail — Lab Table for Selected Patient ──
  if (screen === 'labs_detail') {
    const getFlagColor = (interp: string) => {
      if (interp === 'HH' || interp === 'H') return C.critical.default;
      if (interp === 'LL' || interp === 'L') return C.warning.default;
      return C.success.default;
    };
    const getFlagBg = (interp: string) => {
      if (interp === 'HH' || interp === 'H') return C.critical.bg;
      if (interp === 'LL' || interp === 'L') return C.warning.bg;
      return C.success.bg;
    };

    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={styles.header}>
          <TouchableOpacity onPress={() => setScreen('labs')}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{maskName(selectedPatientName)}</Text>
          <Text style={styles.headerSubtitle}>Lab Results</Text>
        </View>

        <ScrollView style={styles.scrollContent}>
          {loadingLabs ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={C.warning.default} />
              <Text style={styles.loadingText}>Loading lab results...</Text>
            </View>
          ) : labResults.length === 0 ? (
            <View style={{alignItems: 'center', paddingVertical: 40}}>
              <Text style={styles.mutedText}>No lab results found for this patient.</Text>
            </View>
          ) : (
            <>
              <View style={styles.labHeaderRow}>
                <Text style={[styles.labHeaderCell, {flex: 2}]}>Test</Text>
                <Text style={[styles.labHeaderCell, {flex: 1}]}>Value</Text>
                <Text style={[styles.labHeaderCell, {flex: 1}]}>Ref Range</Text>
                <Text style={[styles.labHeaderCell, {width: 40, textAlign: 'center'}]}>Flag</Text>
              </View>

              {labResults.map((lab, idx) => (
                <View key={lab.id || String(idx)} style={styles.labRow}>
                  <View style={{flex: 2}}>
                    <Text style={styles.labTestName}>{lab.loinc_display}</Text>
                    <Text style={styles.labLoincCode}>{lab.loinc_code}</Text>
                  </View>
                  <Text style={[styles.labValue, {flex: 1}]}>
                    {lab.value} {lab.unit}
                  </Text>
                  <Text style={[styles.labRef, {flex: 1}]}>
                    {lab.reference_low != null && lab.reference_high != null
                      ? `${lab.reference_low}-${lab.reference_high}`
                      : '--'}
                  </Text>
                  <View style={[styles.labFlagBadge, {backgroundColor: getFlagBg(lab.interpretation)}]}>
                    <Text style={[styles.labFlagText, {color: getFlagColor(lab.interpretation)}]}>
                      {lab.interpretation}
                    </Text>
                  </View>
                </View>
              ))}
            </>
          )}
        </ScrollView>

        {labResults.length > 0 && (
          <View style={[styles.bottomBar, {flexDirection: 'row'}]}>
            <TouchableOpacity
              style={[styles.primaryButton, {flex: 1, marginRight: 6}]}
              onPress={async () => {
                setEhrNavSession(false);
                await resetChatState();
                const abnormal = labResults.filter(l => l.interpretation !== 'N');
                const summary = abnormal.length > 0
                  ? abnormal.map(l => `${l.loinc_display}: ${l.value} ${l.unit} (${l.interpretation})`).join(', ')
                  : labResults.map(l => `${l.loinc_display}: ${l.value} ${l.unit}`).join(', ');
                sendChatMessage(`Review these lab results for patient ${maskName(selectedPatientName)} and provide clinical interpretation:\n${summary}`);
                setScreen('chat');
              }}>
              <Text style={styles.primaryButtonText}>AI Summary</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.primaryButton, {flex: 1, marginLeft: 6, backgroundColor: '#2E7D32'}]}
              onPress={async () => {
                await resetChatState();
                setEhrNavSession(true);
                setScreen('chat');
              }}>
              <Text style={styles.primaryButtonText}>EHR Navigator</Text>
            </TouchableOpacity>
          </View>
        )}
      </SafeAreaView>
    );
  }

  // ─── Screen: Chat ──────────────────────────────────────
  if (screen === 'chat') {
    const renderMessage = ({item, index}: {item: ChatMessage; index: number}) => {
      const isUser = item.role === 'user';
      const isStreaming = chatGenerating && !isUser && index === displayData.length - 1;
      const hasReasoning = !isUser && item.reasoning && item.reasoning.length > 0;
      const isExpanded = expandedReasoning.has(index);

      return (
        <View style={isUser ? styles.chatRowUser : styles.chatRowAI}>
          {!isUser && (
            <View style={[styles.chatAvatarAI, ehrNavSession && {backgroundColor: '#2E7D32'}]}>
              <Text style={styles.chatAvatarText}>{ehrNavSession ? 'EN' : 'MG'}</Text>
            </View>
          )}
          <View style={{flex: 1}}>
            {hasReasoning && (
              <TouchableOpacity
                style={{
                  backgroundColor: '#1a2a1a',
                  borderRadius: 12,
                  padding: 10,
                  marginBottom: 6,
                  borderWidth: 1,
                  borderColor: '#2E7D32',
                  borderStyle: 'dashed',
                  marginRight: 40,
                }}
                onPress={() => {
                  setExpandedReasoning(prev => {
                    const next = new Set(prev);
                    if (next.has(index)) { next.delete(index); } else { next.add(index); }
                    return next;
                  });
                }}>
                <View style={{flexDirection: 'row', alignItems: 'center', marginBottom: isExpanded ? 6 : 0}}>
                  <Text style={{color: '#4CAF50', fontSize: 12, fontWeight: '600', flex: 1}}>
                    {isExpanded ? '\u25BC' : '\u25B6'} Agent Reasoning
                  </Text>
                  <Text style={{color: '#666', fontSize: 11}}>
                    {isExpanded ? 'tap to collapse' : 'tap to expand'}
                  </Text>
                </View>
                {isExpanded && (
                  <Text style={{color: '#8a9a8a', fontSize: 12.5, lineHeight: 18}}>
                    {item.reasoning}
                  </Text>
                )}
              </TouchableOpacity>
            )}
            <View style={isUser ? styles.chatBubbleUser : styles.chatBubbleAI}>
              {isUser ? (
                <Text style={styles.chatTextUser}>{item.content}</Text>
              ) : (
                <>{renderMarkdown(item.content + (isStreaming ? ' \u2588' : ''), styles.chatTextAI)}</>
              )}
            </View>
          </View>
        </View>
      );
    };

    const displayData: ChatMessage[] = [...chatMessages];
    if (chatGenerating && chatStreamText) {
      displayData.push({role: 'assistant', content: chatStreamText});
    }

    const contextParts: string[] = [];
    if (labResults.length > 0) contextParts.push(`${labResults.length} labs loaded`);
    if (soapNote) contextParts.push('SOAP note loaded');

    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="dark-content" backgroundColor={C.bg.header} />
        <View style={[styles.header, ehrNavSession && {backgroundColor: '#1B5E20'}]}>
          <TouchableOpacity onPress={async () => {
            // Cancel any ongoing generation
            if (chatGenerating && llamaContext) {
              try { await llamaContext.stopCompletion(); } catch {}
            }
            setChatGenerating(false);
            setChatStreamText('');
            setEhrNavigating(false);
            setEhrNavSession(false);
            setScreen('home');
          }}>
            <Text style={styles.backButton}>{'\u2039'} Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{ehrNavSession ? 'EHR Navigator' : 'Clinical Chat'}</Text>
          <Text style={styles.headerSubtitle}>
            {ehrNavSession
              ? `Patient: ${maskName(selectedPatientName)} | Workstation AI`
              : contextParts.length > 0 ? contextParts.join(' | ') : 'Ask any clinical question'}
          </Text>
        </View>

        <KeyboardAvoidingView style={{flex: 1}} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          {displayData.length === 0 && !chatGenerating && !ehrNavigating ? (
            <View style={{flex: 1, justifyContent: 'center', alignItems: 'center', padding: 32}}>
              {ehrNavSession ? (
                <>
                  <Text style={[styles.mutedText, {fontSize: 18, fontWeight: '600', color: '#2E7D32', marginBottom: 12}]}>
                    EHR Navigator Agent
                  </Text>
                  <Text style={[styles.mutedText, {fontSize: 14, textAlign: 'center', lineHeight: 22}]}>
                    Ask any clinical question about{'\n'}{maskName(selectedPatientName)}'s records.{'\n\n'}
                    The workstation agent will search FHIR{'\n'}resources to find the answer.
                  </Text>
                </>
              ) : (
                <Text style={[styles.mutedText, {fontSize: 16, textAlign: 'center'}]}>
                  Ask MedGemma any clinical question.{'\n\n'}
                  {labResults.length > 0 ? 'Lab results are loaded as context.' : 'Load lab results for context-aware answers.'}
                </Text>
              )}
            </View>
          ) : ehrNavigating ? (
            <ScrollView contentContainerStyle={{flexGrow: 1, padding: 16}}>
              {/* User message at top */}
              {displayData.filter(m => m.role === 'user').map((m, i) => (
                <View key={i} style={styles.chatRowUser}>
                  <View style={styles.chatBubbleUser}>
                    <Text style={styles.chatTextUser}>{m.content}</Text>
                  </View>
                </View>
              ))}

              {/* Streaming reasoning box */}
              <View style={styles.chatRowAI}>
                <View style={[styles.chatAvatarAI, {backgroundColor: '#2E7D32'}]}>
                  <Text style={styles.chatAvatarText}>EN</Text>
                </View>
                <View style={{flex: 1}}>
                  <View style={{
                    backgroundColor: '#1a2a1a',
                    borderRadius: 12,
                    padding: 12,
                    borderWidth: 1,
                    borderColor: '#2E7D32',
                    borderStyle: 'dashed',
                  }}>
                    <View style={{flexDirection: 'row', alignItems: 'center', marginBottom: 8}}>
                      <ActivityIndicator size="small" color="#4CAF50" />
                      <Text style={{color: '#4CAF50', fontSize: 13, fontWeight: '600', marginLeft: 8}}>
                        {streamingStep || 'Starting agent...'}
                      </Text>
                    </View>
                    {streamingReasoning ? (
                      <>{renderMarkdown(streamingReasoning, {color: '#8a9a8a', fontSize: 12.5, lineHeight: 18})}</>
                    ) : (
                      <Text style={{color: '#555', fontSize: 12, fontStyle: 'italic'}}>
                        Waiting for data...
                      </Text>
                    )}
                  </View>
                  <Text style={{color: '#555', fontSize: 11, marginTop: 6, marginLeft: 4}}>
                    Workstation AI (LangGraph)
                  </Text>
                </View>
              </View>
            </ScrollView>
          ) : (
            <FlatList
              ref={chatListRef}
              data={displayData}
              renderItem={renderMessage}
              keyExtractor={(_, idx) => String(idx)}
              contentContainerStyle={styles.chatList}
              onContentSizeChange={() => chatListRef.current?.scrollToEnd({animated: true})}
            />
          )}

          {chatGenerating && !chatStreamText && (
            <View style={[styles.loadingContainer, {paddingVertical: 8}]}>
              <ActivityIndicator size="small" color={C.primary.default} />
              <Text style={[styles.loadingText, {fontSize: 13}]}>
                {llamaContext ? 'Generating...' : 'Loading MedGemma...'}
              </Text>
            </View>
          )}

          {chatRecording && (
            <View style={styles.chatRecordingBar}>
              <PulsingDot />
              <Text style={styles.chatRecordingTime}>{recordingTime}</Text>
            </View>
          )}
          {chatTranscribing && (
            <View style={[styles.loadingContainer, {paddingVertical: 6}]}>
              <ActivityIndicator size="small" color={C.warning.default} />
              <Text style={[styles.loadingText, {fontSize: 13}]}>Transcribing with MedASR...</Text>
            </View>
          )}

          {ehrNavigating && displayData.length > 0 && (
            <View style={[styles.loadingContainer, {paddingVertical: 8}]}>
              <ActivityIndicator size="small" color="#2E7D32" />
              <Text style={[styles.loadingText, {fontSize: 13, color: '#2E7D32'}]}>
                EHR Navigator working...
              </Text>
            </View>
          )}

          <View style={[styles.chatInputBar, ehrNavSession && {borderTopColor: '#2E7D32'}]}>
            <TouchableOpacity
              style={[styles.chatMicButton, chatRecording && styles.chatMicButtonActive]}
              disabled={chatGenerating || chatTranscribing || ehrNavigating}
              onPress={toggleChatVoice}>
              <Text style={styles.chatMicIcon}>{chatRecording ? '\u25A0' : '\uD83C\uDFA4'}</Text>
            </TouchableOpacity>
            <TextInput
              style={styles.chatInputFieldFull}
              value={chatInput}
              onChangeText={setChatInput}
              placeholder={chatRecording ? 'Recording...' : ehrNavSession ? `Ask about ${maskName(selectedPatientName)}'s records...` : 'Ask a clinical question...'}
              placeholderTextColor={chatRecording ? C.critical.default : ehrNavSession ? '#2E7D32' : C.text.muted}
              multiline
              maxLength={1000}
              editable={!chatGenerating && !chatRecording && !chatTranscribing && !ehrNavigating}
              returnKeyType="send"
              onSubmitEditing={() => { if (chatInput.trim()) sendChatMessage(chatInput); }}
            />
            {chatGenerating ? (
              <TouchableOpacity
                style={[styles.chatSendButton, {backgroundColor: C.critical.default}]}
                onPress={async () => {
                  if (llamaContext) {
                    await llamaContext.stopCompletion();
                  }
                }}>
                <Text style={styles.chatSendIcon}>{'\u25A0'}</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                style={[styles.chatSendButton, ehrNavSession && {backgroundColor: '#2E7D32'}, !chatInput.trim() && {opacity: 0.4}]}
                disabled={!chatInput.trim() || chatTranscribing || ehrNavigating}
                onPress={() => sendChatMessage(chatInput)}>
                <Text style={styles.chatSendIcon}>{'\u2192'}</Text>
              </TouchableOpacity>
            )}
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  return null;
}

// ─── Styles ──────────────────────────────────────────────
const MONO = Platform.OS === 'android' ? 'monospace' : 'Menlo';

const styles = StyleSheet.create({
  // Layout
  container: {
    flex: 1,
    backgroundColor: C.bg.base,
  },
  centerScreen: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  scrollContent: {
    flex: 1,
    padding: 16,
  },

  // Home
  homeContent: {
    flexGrow: 1,
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'android' ? (StatusBar.currentHeight ?? 24) + 16 : 20,
    paddingBottom: 16,
  },
  homeHeader: {
    alignItems: 'center',
    marginBottom: 32,
  },
  homeGreeting: {
    fontSize: 26,
    fontWeight: '700',
    color: C.text.primary,
    letterSpacing: -0.3,
    marginBottom: 6,
  },
  homeTagline: {
    fontSize: 15,
    color: C.text.secondary,
  },
  homeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    gap: 12,
  },
  homeCard: {
    width: '47%' as any,
    backgroundColor: C.bg.card,
    borderRadius: 16,
    padding: 18,
    borderWidth: 1,
    borderColor: C.border.default,
    alignItems: 'center',
  },
  homeCardIcon: {
    width: 52,
    height: 52,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  homeCardEmoji: {
    fontSize: 24,
  },
  homeCardTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: C.text.primary,
    textAlign: 'center',
  },
  homeCardDesc: {
    fontSize: 12,
    color: C.text.secondary,
    textAlign: 'center',
    marginTop: 4,
  },
  homeModelInfo: {
    fontSize: 12,
    color: C.text.muted,
    textAlign: 'center',
    marginTop: 24,
    lineHeight: 18,
  },
  homeInputBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: C.bg.header,
    borderTopWidth: 1,
    borderTopColor: C.border.default,
  },
  homeInputMic: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: C.primary.soft,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  homeInputMicIcon: {
    fontSize: 20,
  },
  homeInputField: {
    flex: 1,
    backgroundColor: C.bg.elevated,
    borderRadius: 21,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: C.border.default,
    marginRight: 10,
  },
  homeInputPlaceholder: {
    color: C.text.muted,
    fontSize: 14,
  },
  homeInputSend: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: C.primary.default,
    justifyContent: 'center',
    alignItems: 'center',
  },
  homeInputSendIcon: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
  },

  // Header
  header: {
    backgroundColor: C.bg.header,
    paddingHorizontal: 16,
    paddingTop: Platform.OS === 'android' ? (StatusBar.currentHeight ?? 24) + 8 : 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: C.border.default,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: C.text.primary,
  },
  headerSubtitle: {
    fontSize: 12,
    color: C.warning.default,
    marginTop: 3,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
  backButton: {
    color: C.primary.default,
    fontSize: 18,
    fontWeight: '600',
    paddingVertical: 6,
    paddingRight: 16,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  homeButton: {
    color: C.text.muted,
    fontSize: 13,
    fontWeight: '500',
    paddingVertical: 6,
    paddingLeft: 16,
  },

  // Step Indicator
  stepBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    paddingHorizontal: 16,
    backgroundColor: C.bg.header,
    borderBottomWidth: 1,
    borderBottomColor: C.border.default,
  },
  stepItem: {
    alignItems: 'center',
    minWidth: 54,
  },
  stepItemActive: {},
  stepItemDone: {},
  stepNum: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: C.bg.elevated,
    color: C.text.muted,
    fontSize: 12,
    fontWeight: '700',
    textAlign: 'center',
    textAlignVertical: 'center',
    lineHeight: 24,
    overflow: 'hidden',
  },
  stepNumActive: {
    backgroundColor: C.primary.default,
    color: '#fff',
  },
  stepNumDone: {
    backgroundColor: C.success.default,
    color: '#fff',
  },
  stepLabel: {
    fontSize: 10,
    color: C.text.muted,
    marginTop: 3,
    fontWeight: '600',
  },
  stepLabelActive: {
    color: C.primary.default,
  },
  stepLabelDone: {
    color: C.success.default,
  },
  stepConnector: {
    height: 2,
    flex: 1,
    backgroundColor: C.bg.elevated,
    marginHorizontal: 4,
    marginBottom: 14,
  },
  stepConnectorDone: {
    backgroundColor: C.success.default,
  },

  // Recording
  timerText: {
    fontSize: 56,
    fontWeight: '200',
    color: C.text.primary,
    fontVariant: ['tabular-nums'],
    letterSpacing: 2,
    marginBottom: 24,
  },
  pulseRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 32,
    gap: 10,
  },
  pulseDot: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: C.critical.default,
  },
  pulseText: {
    color: C.critical.default,
    fontSize: 16,
    fontWeight: '600',
  },
  circleButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 6,
  },
  circleButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '800',
    letterSpacing: 1,
  },
  recordBtn: {
    backgroundColor: C.critical.default,
  },
  stopBtn: {
    backgroundColor: C.bg.elevated,
    borderWidth: 2,
    borderColor: C.border.default,
  },
  stopSquare: {
    width: 22,
    height: 22,
    borderRadius: 3,
    backgroundColor: C.critical.default,
  },
  audioDurationChip: {
    alignSelf: 'center',
    backgroundColor: C.bg.card,
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 6,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  audioDurationText: {
    color: C.text.secondary,
    fontSize: 13,
    fontWeight: '600',
  },

  // Buttons
  primaryButton: {
    backgroundColor: C.primary.default,
    paddingHorizontal: 28,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    backgroundColor: C.bg.elevated,
    paddingHorizontal: 28,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: C.border.default,
  },
  secondaryButtonText: {
    color: C.text.secondary,
    fontSize: 16,
    fontWeight: '600',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'center',
  },

  // Bottom bar
  bottomBar: {
    padding: 16,
    backgroundColor: C.bg.header,
    borderTopWidth: 1,
    borderTopColor: C.border.default,
  },

  // Section labels
  sectionLabel: {
    color: C.text.secondary,
    fontSize: 11,
    fontWeight: '700',
    marginBottom: 8,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },

  // Cards
  card: {
    backgroundColor: C.bg.card,
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: C.border.default,
  },

  // Inputs
  transcriptInput: {
    backgroundColor: C.bg.elevated,
    borderRadius: 12,
    padding: 16,
    color: C.text.primary,
    fontSize: 15,
    lineHeight: 24,
    minHeight: 200,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  soapInput: {
    backgroundColor: C.bg.elevated,
    borderRadius: 12,
    padding: 16,
    color: C.text.primary,
    fontSize: 14,
    lineHeight: 22,
    minHeight: 400,
    borderWidth: 1,
    borderColor: C.border.default,
    fontFamily: MONO,
  },

  // Streaming
  streamingPreview: {
    marginTop: 12,
    padding: 12,
    backgroundColor: C.primary.soft,
    borderRadius: 10,
    borderLeftWidth: 3,
    borderLeftColor: C.primary.default,
  },
  streamingLabel: {
    color: C.primary.default,
    fontSize: 11,
    fontWeight: '700',
    marginBottom: 4,
    letterSpacing: 0.5,
  },
  streamingText: {
    color: C.text.secondary,
    fontSize: 13,
    lineHeight: 18,
  },

  // Loading
  loadingContainer: {
    alignItems: 'center',
    padding: 16,
  },
  loadingText: {
    color: C.text.secondary,
    fontSize: 14,
    marginTop: 12,
    textAlign: 'center',
    lineHeight: 20,
  },

  // Hints
  hint: {
    color: C.text.muted,
    fontSize: 12,
    marginTop: 32,
    textAlign: 'center',
    lineHeight: 20,
  },

  // Medications pills
  medPillRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  medPill: {
    backgroundColor: C.primary.soft,
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderWidth: 1,
    borderColor: C.primary.default,
  },
  medPillText: {
    color: C.primary.default,
    fontSize: 13,
    fontWeight: '600',
  },

  // Alert cards
  dangerCard: {
    backgroundColor: C.critical.bg,
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: C.critical.border,
    borderLeftWidth: 3,
    borderLeftColor: C.critical.default,
  },
  dangerTitle: {
    color: C.critical.light,
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 10,
  },
  interactionRow: {
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: C.critical.border,
  },
  interactionDrugs: {
    color: C.critical.light,
    fontSize: 14,
    fontWeight: '600',
  },
  interactionType: {
    color: C.text.secondary,
    fontSize: 13,
    marginTop: 2,
  },
  alertWarning: {
    color: C.warning.default,
    fontSize: 12,
    fontWeight: '600',
    marginTop: 12,
    textAlign: 'center',
  },
  safeCard: {
    backgroundColor: C.success.bg,
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: C.success.border,
    borderLeftWidth: 3,
    borderLeftColor: C.success.default,
  },
  safeText: {
    color: C.success.light,
    fontSize: 14,
  },
  mutedText: {
    color: C.text.muted,
    fontSize: 14,
  },

  // ICD-10 codes
  codeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderBottomWidth: 1,
    borderBottomColor: C.border.default,
  },
  codeRowAccepted: {
    backgroundColor: C.success.bg,
  },
  codeInfo: {
    flex: 1,
  },
  codeText: {
    color: C.text.code,
    fontSize: 15,
    fontWeight: '700',
    fontFamily: MONO,
  },
  codeDesc: {
    color: C.text.secondary,
    fontSize: 13,
    marginTop: 2,
  },
  codeToggle: {
    color: C.text.muted,
    fontSize: 18,
    marginLeft: 12,
  },

  // Export / resources
  successBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: C.success.bg,
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: C.success.border,
    gap: 10,
  },
  successIcon: {
    fontSize: 24,
    color: C.success.default,
    fontWeight: '700',
  },
  successText: {
    color: C.success.default,
    fontSize: 16,
    fontWeight: '700',
  },
  resourceCard: {
    backgroundColor: C.bg.card,
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
    borderLeftWidth: 3,
    borderLeftColor: C.primary.default,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  resourceCardDanger: {
    borderLeftColor: C.critical.default,
  },
  resourceType: {
    color: C.warning.default,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  resourceId: {
    color: C.text.code,
    fontSize: 13,
    marginTop: 4,
    fontFamily: MONO,
  },
  resourceDetail: {
    color: C.text.secondary,
    fontSize: 13,
    marginTop: 2,
  },
  errorCard: {
    backgroundColor: C.critical.bg,
    borderRadius: 10,
    padding: 12,
    marginTop: 12,
    borderWidth: 1,
    borderColor: C.critical.border,
  },
  errorTitle: {
    color: C.critical.light,
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 8,
  },
  errorText: {
    color: C.critical.light,
    fontSize: 12,
    marginTop: 4,
  },
  dashboardLink: {
    color: C.text.muted,
    fontSize: 12,
    textAlign: 'center',
    marginTop: 20,
    marginBottom: 20,
  },

  // Radiology
  imagePicker: {
    backgroundColor: C.bg.card,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: C.border.default,
    borderStyle: 'dashed',
    overflow: 'hidden',
    minHeight: 200,
  },
  imagePreview: {
    width: '100%',
    height: 250,
  },
  imagePickerPlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  imagePickerIcon: {
    fontSize: 48,
    color: C.accent.default,
    fontWeight: '300',
  },
  imagePickerText: {
    color: C.text.muted,
    fontSize: 14,
    marginTop: 8,
  },
  typeRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  typePill: {
    paddingHorizontal: 16,
    paddingVertical: 9,
    borderRadius: 20,
    backgroundColor: C.bg.card,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  typePillActive: {
    backgroundColor: C.accent.default,
    borderColor: C.accent.default,
  },
  typePillText: {
    color: C.text.secondary,
    fontSize: 13,
    fontWeight: '600',
  },
  typePillTextActive: {
    color: '#fff',
  },
  contextInput: {
    backgroundColor: C.bg.elevated,
    borderRadius: 12,
    padding: 14,
    color: C.text.primary,
    fontSize: 14,
    lineHeight: 20,
    minHeight: 60,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  reportThumbnail: {
    width: '100%',
    height: 180,
    borderRadius: 12,
    marginBottom: 16,
    backgroundColor: C.bg.card,
  },
  reportText: {
    color: C.text.primary,
    fontSize: 14,
    lineHeight: 22,
  },
  disclaimerCard: {
    backgroundColor: C.warning.bg,
    borderRadius: 12,
    padding: 14,
    marginTop: 20,
    borderWidth: 1,
    borderColor: C.warning.border,
  },
  disclaimerText: {
    color: C.warning.default,
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
    lineHeight: 18,
  },

  // ─── Labs ──────────────────────────────────────────
  labHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: C.border.default,
    marginBottom: 4,
  },
  labHeaderCell: {
    color: C.text.muted,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  labRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: C.border.subtle,
  },
  labTestName: {
    color: C.text.primary,
    fontSize: 14,
    fontWeight: '600',
  },
  labLoincCode: {
    color: C.text.muted,
    fontSize: 11,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  labValue: {
    color: C.text.primary,
    fontSize: 14,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  labRef: {
    color: C.text.muted,
    fontSize: 12,
  },
  labFlagBadge: {
    width: 40,
    height: 24,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  labFlagText: {
    fontSize: 12,
    fontWeight: '800',
  },

  // ─── Dictation History ───────────────────────────
  dictationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: C.bg.card,
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  dictationTime: {
    color: C.text.secondary,
    fontSize: 12,
    fontWeight: '600',
  },
  dictationPreview: {
    color: C.text.muted,
    fontSize: 12,
    marginTop: 2,
  },
  dictationBadges: {
    flexDirection: 'row',
    gap: 6,
    marginLeft: 8,
  },
  dictationBadge: {
    paddingHorizontal: 6,
    paddingVertical: 3,
    borderRadius: 4,
    borderWidth: 1,
  },
  dictationBadgeText: {
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.5,
  },

  // ─── Patient List ────────────────────────────────
  patientRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: C.bg.card,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  patientInfo: {
    flex: 1,
    marginRight: 12,
  },
  patientName: {
    color: C.text.primary,
    fontSize: 15,
    fontWeight: '600',
  },
  patientMeta: {
    color: C.text.muted,
    fontSize: 12,
    marginTop: 3,
  },
  patientActions: {
    flexDirection: 'row',
    gap: 8,
  },
  patientActionBtn: {
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: C.border.default,
    backgroundColor: C.bg.elevated,
  },
  patientActionText: {
    color: C.text.secondary,
    fontSize: 12,
    fontWeight: '600',
  },

  // ─── Chat ─────────────────────────────────────────
  chatSendButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: C.primary.default,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chatSendIcon: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  chatList: {
    padding: 16,
    paddingBottom: 8,
  },
  chatRowUser: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginBottom: 12,
  },
  chatRowAI: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  chatAvatarAI: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: C.primary.soft,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
    marginTop: 2,
  },
  chatAvatarText: {
    color: C.primary.default,
    fontSize: 11,
    fontWeight: '800',
  },
  chatBubbleUser: {
    maxWidth: '80%',
    backgroundColor: C.primary.soft,
    borderRadius: 18,
    borderBottomRightRadius: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  chatBubbleAI: {
    flex: 1,
    paddingTop: 4,
  },
  chatTextUser: {
    color: C.text.primary,
    fontSize: 15,
    lineHeight: 22,
  },
  chatTextAI: {
    color: C.text.primary,
    fontSize: 15,
    lineHeight: 23,
  },
  chatInputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: 10,
    paddingHorizontal: 12,
    backgroundColor: C.bg.header,
    borderTopWidth: 1,
    borderTopColor: C.border.default,
  },
  chatInputFieldFull: {
    flex: 1,
    backgroundColor: C.bg.elevated,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    color: C.text.primary,
    fontSize: 14,
    maxHeight: 100,
    borderWidth: 1,
    borderColor: C.border.default,
    marginRight: 8,
  },
  chatMicButton: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: C.bg.elevated,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
    borderWidth: 1,
    borderColor: C.border.default,
  },
  chatMicButtonActive: {
    backgroundColor: C.critical.bg,
    borderColor: C.critical.default,
  },
  chatMicIcon: {
    fontSize: 18,
  },
  chatRecordingBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 6,
    gap: 12,
  },
  chatRecordingTime: {
    color: C.critical.default,
    fontSize: 14,
    fontWeight: '600',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
});
