export type TextSource = {
  source_path: string | null;
  page_number: number | null;
  score: number | null;
};

export type VisualEvidence = {
  used: boolean;
  source_path: string | null;
  image_url: string | null;
  entity: string | null;
  score: number | null;
};

export type AskResponse = {
  question: string;
  answer: string;
  explanation: string;
  raw_response: string;
  model: string | null;
  visual: VisualEvidence;
  text_sources: TextSource[];
};