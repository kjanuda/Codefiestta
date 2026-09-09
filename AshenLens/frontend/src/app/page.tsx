"use client";

import { FormEvent, useState } from "react";

import type {
  AskResponse,
} from "@/types/ashenlens";


const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  "http://127.0.0.1:8000";


const SAMPLE_QUESTIONS = [
  "What is the central emblem on the banner of House Morvain?",
  "In the portrait of Ignatz Ashgrove the Oathless, what object are they holding?",
  "According to the official threat-classification plate, what numerical rating is assigned to the creature known as the Weeping Lurker?",
  "According to the figure plate detailing weapon binding, how many shards of will are required to attune The Thrice-Bound Edge?",
];


function cleanSourceName(
  path: string | null,
) {
  if (!path) return "Unknown source";

  return path
    .replaceAll("\\", "/")
    .split("/")
    .pop()
    ?.replace(/\.(pdf|docx|md|txt)$/i, "")
    .replaceAll("_", " ") ?? path;
}


export default function Home() {
  const [question, setQuestion] =
    useState("");

  const [result, setResult] =
    useState<AskResponse | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);


  async function ask(
    selectedQuestion?: string,
  ) {
    const finalQuestion = (
      selectedQuestion ?? question
    ).trim();

    if (!finalQuestion || loading) {
      return;
    }

    setQuestion(finalQuestion);
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/api/ask`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            question: finalQuestion,
          }),
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ??
            "Tensor X could not answer the question.",
        );
      }

      setResult(
        data as AskResponse,
      );
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Something went wrong.";

      setError(message);
    } finally {
      setLoading(false);
    }
  }


  function handleSubmit(
    event: FormEvent,
  ) {
    event.preventDefault();

    void ask();
  }


  const imageUrl =
    result?.visual.used &&
    result.visual.image_url
      ? `${API_URL}${result.visual.image_url}`
      : null;


  return (
    <main className="min-h-screen bg-[#f7f5ef] text-[#171713]">
      {/* Header */}
      <header className="border-b border-black/10 bg-[#f7f5ef]/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-5 md:px-8">
          <div>
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full border border-black/15 bg-[#191914] text-sm font-semibold text-white">
                T
              </div>

              <div>
                <h1 className="text-lg font-semibold tracking-[-0.02em]">
                  Tensor X
                </h1>

                <p className="text-xs text-black/45">
                  Multimodal Archive Intelligence
                </p>
              </div>
            </div>
          </div>

          <div className="hidden items-center gap-2 rounded-full border border-black/10 bg-white/60 px-3 py-1.5 text-xs text-black/55 sm:flex">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />

            Tensor X Archive
          </div>
        </div>
      </header>


      <section className="mx-auto max-w-7xl px-5 py-12 md:px-8 md:py-16">
        {/* Intro */}
        <div className="max-w-3xl">
          <p className="mb-3 text-xs font-medium uppercase tracking-[0.18em] text-black/45">
            Intelligent Document Assistant
          </p>

          <h2 className="max-w-2xl text-4xl font-medium leading-[1.08] tracking-[-0.045em] md:text-5xl">
            Ask the archive.
            <br />
            See the evidence.
          </h2>

          <p className="mt-5 max-w-xl text-sm leading-6 text-black/55 md:text-base">
            Tensor X retrieves textual evidence,
            identifies relevant visual material,
            and answers using the archive itself.
          </p>
        </div>


        {/* Search */}
        <form
          onSubmit={handleSubmit}
          className="mt-10 max-w-4xl"
        >
          <div className="rounded-2xl border border-black/10 bg-white p-2 shadow-[0_16px_50px_rgba(0,0,0,0.05)]">
            <textarea
              value={question}
              onChange={(event) =>
                setQuestion(
                  event.target.value,
                )
              }
              placeholder="Ask a question about the Tensor X Archive..."
              rows={3}
              className="w-full resize-none bg-transparent px-4 py-3 text-[15px] leading-6 outline-none placeholder:text-black/30"
            />

            <div className="flex items-center justify-between border-t border-black/5 px-2 pt-2">
              <div className="px-2 text-xs text-black/35">
                Visual + semantic retrieval
              </div>

              <button
                type="submit"
                disabled={
                  loading ||
                  !question.trim()
                }
                className="rounded-xl bg-[#191914] px-5 py-2.5 text-sm font-medium text-white transition hover:bg-black disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading
                  ? "Searching..."
                  : "Ask Tensor X"}
              </button>
            </div>
          </div>
        </form>


        {/* Samples */}
        {!result && !loading && (
          <div className="mt-7">
            <p className="mb-3 text-xs text-black/40">
              Try an archive question
            </p>

            <div className="flex max-w-5xl flex-wrap gap-2">
              {SAMPLE_QUESTIONS.map(
                (sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() =>
                      void ask(sample)
                    }
                    className="max-w-full rounded-full border border-black/10 bg-white/50 px-4 py-2 text-left text-xs leading-5 text-black/60 transition hover:border-black/20 hover:bg-white hover:text-black"
                  >
                    {sample}
                  </button>
                ),
              )}
            </div>
          </div>
        )}


        {/* Loading */}
        {loading && (
          <div className="mt-12 max-w-5xl rounded-2xl border border-black/10 bg-white p-6">
            <div className="flex items-center gap-3">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-black/15 border-t-black" />

              <div>
                <p className="text-sm font-medium">
                  Searching the archive
                </p>

                <p className="mt-1 text-xs text-black/45">
                  Retrieving text,
                  ranking evidence and
                  inspecting relevant visuals.
                </p>
              </div>
            </div>
          </div>
        )}


        {/* Error */}
        {error && (
          <div className="mt-10 max-w-4xl rounded-2xl border border-red-200 bg-red-50 p-5">
            <p className="text-sm font-medium text-red-900">
              Unable to complete request
            </p>

            <p className="mt-2 text-sm text-red-700">
              {error}
            </p>
          </div>
        )}


        {/* Result */}
        {result && !loading && (
          <div className="mt-12 grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(340px,0.85fr)]">
            {/* Answer */}
            <section className="rounded-3xl border border-black/10 bg-white p-6 shadow-[0_16px_50px_rgba(0,0,0,0.04)] md:p-8">
              <div className="flex items-center justify-between gap-4">
                <p className="text-xs font-medium uppercase tracking-[0.16em] text-black/40">
                  Answer
                </p>

                {result.model && (
                  <span className="rounded-full bg-black/[0.04] px-3 py-1 text-[11px] text-black/45">
                    {result.model}
                  </span>
                )}
              </div>

              <h3 className="mt-5 text-4xl font-medium tracking-[-0.04em] md:text-5xl">
                {result.answer}
              </h3>

              <p className="mt-6 max-w-2xl text-[15px] leading-7 text-black/65">
                {result.explanation}
              </p>


              {/* Sources */}
              <div className="mt-9 border-t border-black/10 pt-6">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-medium uppercase tracking-[0.14em] text-black/40">
                    Retrieved sources
                  </p>

                  <span className="text-xs text-black/35">
                    {result.text_sources.length} passages
                  </span>
                </div>

                <div className="mt-4 space-y-2">
                  {result.text_sources.map(
                    (source, index) => (
                      <div
                        key={`${source.source_path}-${index}`}
                        className="rounded-xl border border-black/[0.07] bg-[#faf9f5] px-4 py-3"
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium">
                              {cleanSourceName(
                                source.source_path,
                              )}
                            </p>

                            <p className="mt-1 truncate text-xs text-black/40">
                              {source.source_path}
                            </p>
                          </div>

                          {source.score !==
                            null && (
                            <span className="shrink-0 text-xs tabular-nums text-black/35">
                              {source.score.toFixed(
                                3,
                              )}
                            </span>
                          )}
                        </div>

                        {source.page_number !==
                          null && (
                          <p className="mt-2 text-[11px] text-black/35">
                            Page{" "}
                            {source.page_number}
                          </p>
                        )}
                      </div>
                    ),
                  )}
                </div>
              </div>
            </section>


            {/* Visual Evidence */}
            <aside className="rounded-3xl border border-black/10 bg-[#1a1a16] p-3 text-white shadow-[0_16px_50px_rgba(0,0,0,0.08)]">
              {imageUrl ? (
                <>
                  <div className="overflow-hidden rounded-[18px] bg-black/20">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={imageUrl}
                      alt={
                        result.visual.entity ??
                        "Retrieved archive visual"
                      }
                      className="aspect-[4/3] w-full object-contain"
                    />
                  </div>

                  <div className="px-3 pb-3 pt-5">
                    <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-white/40">
                      Visual evidence
                    </p>

                    <h4 className="mt-2 text-lg font-medium tracking-[-0.02em]">
                      {result.visual.entity ??
                        "Archive visual"}
                    </h4>

                    <p className="mt-2 break-all text-xs leading-5 text-white/45">
                      {
                        result.visual
                          .source_path
                      }
                    </p>

                    {result.visual.score !==
                      null && (
                      <div className="mt-4 flex items-center justify-between border-t border-white/10 pt-3 text-xs">
                        <span className="text-white/40">
                          Retrieval confidence
                        </span>

                        <span className="font-medium tabular-nums text-white/75">
                          {result.visual.score.toFixed(
                            2,
                          )}
                        </span>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="flex min-h-[320px] items-center justify-center p-8 text-center text-sm text-white/45">
                  No visual evidence was
                  required for this answer.
                </div>
              )}
            </aside>
          </div>
        )}
      </section>


      <footer className="mx-auto max-w-7xl border-t border-black/10 px-5 py-6 text-xs text-black/35 md:px-8">
        Tensor X · Grounded multimodal
        retrieval for the archive
      </footer>
    </main>
  );
}