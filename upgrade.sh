#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║         🎌 ANIBOT — FULL SYSTEM UPGRADE v3.0                            ║
# ║  Backend  : FastAPI + RAG + Anime Card + Rating + Smart Context         ║
# ║  Frontend : Dark Glassmorphism UI + Slider + Table + Routing            ║
# ║  Dataset  : github.com/InDsToReE/Data-Anime-Json                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# CARA PAKE:
#   chmod +x upgrade.sh && ./upgrade.sh

set -euo pipefail
DIR="/root/anime-ai-chat"
BACK="$DIR/backend"
FRONT="$DIR/frontend"
DATA="$DIR/data"

G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; P='\033[0;35m'; W='\033[1;37m'; N='\033[0m'
ok()   { printf "${G}  ✓ %s${N}\n" "$1"; }
info() { printf "${C}  → %s${N}\n" "$1"; }
step() { printf "\n${P}${W}━━━ %s ━━━${N}\n" "$1"; }

# ─────────────────────────────────────────
# 1. UPDATE DATASET
# ─────────────────────────────────────────
step "Update Dataset dari GitHub"
info "Download anime.json..."
curl -fsSL "https://raw.githubusercontent.com/InDsToReE/Data-Anime-Json/refs/heads/main/anime.json" -o "$DATA/anime.json" && ok "anime.json updated"
info "Download home.json..."
curl -fsSL "https://raw.githubusercontent.com/InDsToReE/Data-Anime-Json/refs/heads/main/home.json" -o "$DATA/home.json" && ok "home.json updated"

# ─────────────────────────────────────────
# 2. TULIS ULANG BACKEND main.py
# ─────────────────────────────────────────
step "Generate Backend v3.0"
cat > "$BACK/main.py" << 'PYEOF'
import json, os, re, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL    = os.getenv("OLLAMA_MODEL", "qwen2:0.5b")
TOP_K    = 3
MAX_CTX  = 900

app = FastAPI(title="AniBot API v3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    response_time: float
    dataset_detail: int
    dataset_list: int
    anime_card: dict | None = None
    anime_slider: list[dict] = []

# ── helpers ──
def _genres(a):
    return [g["title"] if isinstance(g, dict) else g for g in a.get("genres", [])]

def _ep_total(a):
    ep = a.get("episode_summary", {})
    if isinstance(ep, dict): return ep.get("total", len(a.get("episodes", [])) or "?")
    return len(a.get("episodes", [])) or "?"

def _characters(a):
    out = []
    for c in a.get("characters", [])[:4]:
        if not isinstance(c, dict): continue
        va = c.get("voice_actor", {})
        va_name = va.get("name", "") if isinstance(va, dict) else str(va)
        out.append({"name": c.get("name", ""), "role": c.get("role", ""), "cv": va_name, "image": c.get("image", "")})
    return out

def _recs(a):
    out = []
    for r in a.get("recommendations", [])[:4]:
        if isinstance(r, dict): out.append({"title": r.get("title",""), "slug": r.get("slug",""), "image": r.get("image",""), "status": r.get("status","")})
        else: out.append({"title": str(r), "slug": "", "image": "", "status": ""})
    return out

def _make_card(a):
    return {
        "title":       a.get("title", ""),
        "title_alt":   a.get("title_alternative", ""),
        "slug":        a.get("slug", ""),
        "image":       a.get("image", ""),
        "status":      a.get("status", ""),
        "type":        a.get("type", ""),
        "studio":      a.get("studio", ""),
        "director":    a.get("director", ""),
        "producers":   a.get("producers", ""),
        "season":      a.get("season", a.get("released", "")),
        "released":    a.get("released", ""),
        "genres":      _genres(a),
        "episodes":    _ep_total(a),
        "rating":      a.get("rating"),
        "synopsis":    a.get("synopsis", ""),
        "characters":  _characters(a),
        "recommendations": _recs(a),
        "url":         a.get("url", ""),
    }

def _make_slide(a):
    return {
        "title":  a.get("title", ""),
        "slug":   a.get("slug", ""),
        "image":  a.get("image", ""),
        "status": a.get("status", ""),
        "type":   a.get("type", ""),
        "rating": a.get("rating"),
        "episode": a.get("episode", _ep_total(a)),
    }

class AnimeRAG:
    def __init__(self):
        self.detail:  list[dict] = []
        self.listing: list[dict] = []
        self.idx_d:   list[str]  = []
        self.idx_l:   list[str]  = []
        self._load()

    def _load_json(self, path: Path):
        if not path.exists(): return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list): return raw
            if isinstance(raw, dict):
                for k in ["data","anime","results","items","list"]:
                    if k in raw and isinstance(raw[k], list): return raw[k]
                return [raw]
        except Exception as e:
            print(f"Load error {path.name}: {e}")
        return []

    def _flat(self, a: dict) -> str:
        g = " ".join(_genres(a))
        tags = " ".join(t.get("name","") if isinstance(t,dict) else str(t) for t in a.get("tags",[]))
        return " ".join(filter(None, [
            a.get("title",""), a.get("title_alternative",""), a.get("synopsis",""),
            a.get("studio",""), a.get("director",""), a.get("season",""),
            a.get("status",""), a.get("type",""), a.get("released",""), g, tags
        ])).lower()

    def _load(self):
        for e in self._load_json(DATA_DIR / "anime.json"):
            if isinstance(e, dict) and e.get("title"):
                self.detail.append(e); self.idx_d.append(self._flat(e))
        for e in self._load_json(DATA_DIR / "home.json"):
            if isinstance(e, dict) and e.get("title"):
                self.listing.append(e); self.idx_l.append(self._flat(e))
        print(f"✓ anime.json={len(self.detail)} | home.json={len(self.listing)}")

    def _score(self, kws, text, title):
        s = 0
        for k in kws:
            if len(k) < 2: continue
            cnt = text.count(k)
            s += cnt * (4 if k in title.lower() else 1)
        return s

    def search(self, q: str):
        kws = re.findall(r'\w+', q.lower())
        def top(lst, idx):
            scored = sorted(enumerate(idx), key=lambda x: self._score(kws, x[1], lst[x[0]].get("title","")), reverse=True)
            return [lst[i] for i,_ in scored[:TOP_K] if self._score(kws, idx[i], lst[i].get("title","")) > 0]
        return top(self.detail, self.idx_d), top(self.listing, self.idx_l)

    def top_rated(self, genre: str = "", limit: int = 6):
        pool = [a for a in self.detail if a.get("rating") is not None]
        if genre:
            pool = [a for a in pool if genre.lower() in " ".join(_genres(a)).lower()]
        pool.sort(key=lambda a: float(a.get("rating") or 0), reverse=True)
        return pool[:limit]

    def build_context(self, detail, listing):
        parts = []; total = 0
        for a in detail:
            g = ", ".join(_genres(a))
            ep = _ep_total(a)
            syn = (a.get("synopsis") or "")[:200]
            rating = a.get("rating", "-")
            block = (
                f"Judul: {a.get('title')}\n"
                f"Studio: {a.get('studio','-')} | Tayang: {a.get('season',a.get('released','-'))} | "
                f"Status: {a.get('status','-')} | Ep: {ep} | Rating: {rating}\n"
                f"Genre: {g or '-'}\n"
                f"Sinopsis: {syn}\n"
            )
            if total + len(block) > MAX_CTX: break
            parts.append(block); total += len(block)
        if not detail and listing:
            lines = ["Anime ditemukan:"]
            for a in listing[:5]:
                lines.append(f"- {a.get('title')} ({a.get('status')}, Ep:{a.get('episode','-')})")
            parts.append("\n".join(lines))
        return "\n---\n".join(parts)

rag = AnimeRAG()

SYSTEM = """Kamu AniBot, asisten anime gaul yang informatif dan natural.

BAHASA: Indonesia santai (gue/lo/bro/sis). DILARANG bahasa Inggris.

ATURAN JAWAB:
- Gunakan HANYA data dari KONTEKS. Jangan mengarang tahun, studio, atau rating.
- Jawab sesuai yang ditanya saja — jangan dump semua data.
- Kalau ditanya sinopsis → jelaskan ceritanya dengan kalimat natural.
- Kalau ditanya episode → sebut angkanya saja.
- Kalau ditanya rekomendasi → kasih judul + alasan singkat.
- Kalau topik umum → jawab dari pengetahuan umum.
- Jawaban padat 2-5 kalimat, hindari spasi berlebihan.
- JANGAN format [DETAIL]/[LIST] ke dalam jawaban."""

@app.get("/")
def root():
    return {"status": "AniBot v3.0", "detail": len(rag.detail), "listing": len(rag.listing), "model": MODEL}

@app.get("/stats")
def stats():
    statuses = {}; types = {}; genres_c = {}; studios = set()
    for a in rag.detail:
        s = a.get("status","?"); statuses[s] = statuses.get(s,0)+1
        t = a.get("type","?");   types[t]    = types.get(t,0)+1
        if a.get("studio"): studios.add(a["studio"])
        for g in a.get("genres",[]):
            n = g["title"] if isinstance(g,dict) else g
            genres_c[n] = genres_c.get(n,0)+1
    for a in rag.listing:
        s = a.get("status","?"); statuses[s] = statuses.get(s,0)+1
    top_g = sorted(genres_c.items(), key=lambda x:x[1], reverse=True)[:10]
    return {
        "detail_count": len(rag.detail), "listing_count": len(rag.listing),
        "total": len(rag.detail)+len(rag.listing), "studios": len(studios),
        "statuses": statuses, "types": types, "model": MODEL,
        "top_genres": [{"genre":k,"count":v} for k,v in top_g],
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip(): raise HTTPException(400, "Pesan kosong!")
    t0 = time.time()
    q = req.message

    top_d, top_l = rag.search(q)
    ctx = rag.build_context(top_d, top_l)
    sources = list({a.get("title","?") for a in top_d+top_l})

    # Tentukan apakah perlu kirim anime_card / slider
    anime_card = None
    anime_slider = []

    # Intent detection
    is_detail   = bool(re.search(r'jelasin|ceritain|tentang|detail|info|sinopsis|review|apa itu|apa sih', q.lower()))
    is_list     = bool(re.search(r'list|daftar|rekomend|terbaik|bagus|top|populer|genre|action|romance|isekai|horror|comedy', q.lower()))
    is_episode  = bool(re.search(r'episode|eps|ep\b|berapa ep', q.lower()))

    if top_d:
        if is_detail or is_episode:
            anime_card = _make_card(top_d[0])
        if is_list:
            anime_slider = [_make_slide(a) for a in (top_d + top_l)[:8]]
    elif top_l and is_list:
        anime_slider = [_make_slide(a) for a in top_l[:8]]

    # Kalau minta rekomendasi, ambil top rated juga
    if is_list and not anime_slider:
        rated = rag.top_rated(limit=8)
        anime_slider = [_make_slide(a) for a in rated]

    sys_content = f"{SYSTEM}\n\nKONTEKS:\n{ctx}" if ctx else SYSTEM
    messages = [{"role":"system","content":sys_content}]
    for m in req.history[-4:]:
        if m.get("role") in ("user","assistant") and m.get("content"):
            messages.append({"role":m["role"],"content":m["content"]})
    messages.append({"role":"user","content":q})

    try:
        resp = ollama.chat(model=MODEL, messages=messages,
            options={"temperature":0.65,"num_predict":280,"num_ctx":768,"num_thread":4,"repeat_penalty":1.15})
        reply = resp["message"]["content"]
    except Exception as e:
        if "refused" in str(e).lower(): raise HTTPException(503,"Ollama belum jalan!")
        raise HTTPException(500,str(e))

    return ChatResponse(
        reply=reply, sources=sources,
        response_time=round(time.time()-t0,2),
        dataset_detail=len(rag.detail), dataset_list=len(rag.listing),
        anime_card=anime_card, anime_slider=anime_slider,
    )

@app.post("/reload")
def reload():
    global rag; rag = AnimeRAG()
    return {"ok":True,"detail":len(rag.detail),"listing":len(rag.listing)}

@app.get("/search")
def search(q:str):
    d,l = rag.search(q)
    return {"detail":[_make_card(a) for a in d],"listing":[_make_slide(a) for a in l]}

@app.get("/top-rated")
def top_rated(genre:str="",limit:int=12):
    return [_make_card(a) for a in rag.top_rated(genre,limit)]
PYEOF
ok "backend/main.py ditulis ulang"

# ─────────────────────────────────────────
# 3. TULIS ULANG FRONTEND App.jsx
# ─────────────────────────────────────────
step "Generate Frontend v3.0 (Dark Glassmorphism)"
cat > "$FRONT/src/App.jsx" << 'JSEOF'
import { useState, useRef, useEffect, useCallback } from "react";

const API   = "http://137.184.60.192:8000";
const BASE  = "https://animku.vercel.app";

const SUGGESTIONS = [
  "Jelasin anime Medalist dong",
  "Top anime action terbaik?",
  "Rekomendasiin anime romance rating tinggi",
  "Jujutsu Kaisen itu ceritanya gimana?",
  "Berapa episode One Piece sekarang?",
  "Anime isekai terbaik apa aja?",
];

const ICONS = {
  star:    "https://api.iconify.design/solar:star-bold.svg?color=%23fbbf24",
  tv:      "https://api.iconify.design/solar:tv-bold.svg?color=%238b5cf6",
  play:    "https://api.iconify.design/solar:play-bold.svg?color=white",
  info:    "https://api.iconify.design/solar:info-circle-bold.svg?color=%238b5cf6",
  studio:  "https://api.iconify.design/solar:camera-bold.svg?color=%2394a3b8",
  cal:     "https://api.iconify.design/solar:calendar-bold.svg?color=%2394a3b8",
  ep:      "https://api.iconify.design/solar:list-bold.svg?color=%2394a3b8",
  person:  "https://api.iconify.design/solar:user-bold.svg?color=%2394a3b8",
  genre:   "https://api.iconify.design/solar:tag-bold.svg?color=%2394a3b8",
  recs:    "https://api.iconify.design/solar:bookmark-bold.svg?color=%2394a3b8",
  send:    "https://api.iconify.design/solar:arrow-up-bold.svg?color=white",
  menu:    "https://api.iconify.design/solar:hamburger-menu-bold.svg?color=%2394a3b8",
  close:   "https://api.iconify.design/solar:close-bold.svg?color=%2394a3b8",
  status:  "https://api.iconify.design/solar:shield-check-bold.svg?color=%2334d399",
  reset:   "https://api.iconify.design/solar:restart-bold.svg?color=%2394a3b8",
  prev:    "https://api.iconify.design/solar:arrow-left-bold.svg?color=white",
  next:    "https://api.iconify.design/solar:arrow-right-bold.svg?color=white",
};

function Img({ src, alt, style, fallback = null }) {
  const [err, setErr] = useState(false);
  if (err) return fallback || <div style={{ ...style, background: "rgba(139,92,246,.15)", display:"flex",alignItems:"center",justifyContent:"center", fontSize:11, color:"#475569" }}>No Image</div>;
  return <img src={src} alt={alt} style={style} onError={() => setErr(true)} />;
}

function Icon({ src, size = 16, style = {} }) {
  return <img src={src} alt="" width={size} height={size} style={{ display:"inline-block", verticalAlign:"middle", ...style }} />;
}

// ── Anime Detail Card ──
function AnimeCard({ card }) {
  if (!card) return null;
  const slug = card.slug || card.title?.toLowerCase().replace(/\s+/g,"-");
  const watchUrl   = `${BASE}/watch/${slug}`;
  const detailUrl  = `${BASE}/detail/${slug}`;

  return (
    <div style={{
      margin: "12px 0",
      background: "linear-gradient(135deg,rgba(139,92,246,.12),rgba(99,102,241,.07))",
      border: "1px solid rgba(139,92,246,.25)",
      borderRadius: 16, overflow: "hidden",
    }}>
      {/* Top: poster + info */}
      <div style={{ display:"flex", gap:0 }}>
        {card.image && (
          <div style={{ flexShrink:0, width:110, position:"relative" }}>
            <Img src={card.image} alt={card.title} style={{ width:"100%", height:155, objectFit:"cover", display:"block" }}/>
            <div style={{ position:"absolute",inset:0, background:"linear-gradient(to right,transparent 70%,rgba(7,7,15,.9))" }}/>
          </div>
        )}
        <div style={{ flex:1, padding:"14px 16px 10px", minWidth:0 }}>
          <div style={{ fontWeight:800, fontSize:15, color:"#c4b5fd", marginBottom:4, lineHeight:1.3 }}>{card.title}</div>
          {card.title_alt && <div style={{ fontSize:11, color:"#64748b", marginBottom:8 }}>{card.title_alt}</div>}

          <div style={{ display:"flex", flexWrap:"wrap", gap:5, marginBottom:8 }}>
            {card.rating && (
              <span style={{ display:"flex",alignItems:"center",gap:4, fontSize:12, fontWeight:700, color:"#fbbf24", background:"rgba(251,191,36,.12)", border:"1px solid rgba(251,191,36,.25)", borderRadius:8, padding:"3px 9px" }}>
                <Icon src={ICONS.star} size={13}/> {card.rating}
              </span>
            )}
            {card.status && (
              <span style={{ fontSize:11, padding:"3px 9px", borderRadius:8, fontWeight:600, background: card.status==="Ongoing"?"rgba(52,211,153,.12)":"rgba(99,102,241,.12)", color:card.status==="Ongoing"?"#34d399":"#818cf8", border:`1px solid ${card.status==="Ongoing"?"rgba(52,211,153,.25)":"rgba(99,102,241,.25)"}` }}>
                {card.status}
              </span>
            )}
            {card.type && <span style={{ fontSize:11, padding:"3px 9px", borderRadius:8, background:"rgba(255,255,255,.05)", color:"#94a3b8", border:"1px solid rgba(255,255,255,.08)" }}>{card.type}</span>}
          </div>

          <div style={{ display:"flex", flexDirection:"column", gap:4, fontSize:12, color:"#94a3b8" }}>
            {card.studio && <div style={{ display:"flex",alignItems:"center",gap:6 }}><Icon src={ICONS.studio} size={13}/> {card.studio}</div>}
            {card.season && <div style={{ display:"flex",alignItems:"center",gap:6 }}><Icon src={ICONS.cal} size={13}/> {card.season}</div>}
            {card.episodes && <div style={{ display:"flex",alignItems:"center",gap:6 }}><Icon src={ICONS.ep} size={13}/> {card.episodes} Episode</div>}
          </div>
        </div>
      </div>

      {/* Genres */}
      {card.genres?.length > 0 && (
        <div style={{ padding:"8px 14px", borderTop:"1px solid rgba(255,255,255,.05)", display:"flex", flexWrap:"wrap", gap:5 }}>
          <Icon src={ICONS.genre} size={12} style={{ marginTop:2 }}/>
          {card.genres.map(g => {
            const gSlug = g.toLowerCase().replace(/\s+/g,"-");
            return (
              <a key={g} href={`${BASE}/genres/${gSlug}/1`} target="_blank" rel="noreferrer" style={{
                fontSize:11, padding:"2px 9px", borderRadius:20,
                background:"rgba(139,92,246,.1)", color:"#a78bfa",
                border:"1px solid rgba(139,92,246,.2)", textDecoration:"none",
                transition:"all .15s", cursor:"pointer"
              }}
              onMouseEnter={e=>{e.target.style.background="rgba(139,92,246,.25)";}}
              onMouseLeave={e=>{e.target.style.background="rgba(139,92,246,.1)";}}
              >{g}</a>
            );
          })}
        </div>
      )}

      {/* Characters */}
      {card.characters?.length > 0 && (
        <div style={{ padding:"8px 14px", borderTop:"1px solid rgba(255,255,255,.05)" }}>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:6, display:"flex",alignItems:"center",gap:5 }}>
            <Icon src={ICONS.person} size={12}/> Karakter Utama
          </div>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
            {card.characters.map((c,i) => (
              <div key={i} style={{ display:"flex",alignItems:"center",gap:6, background:"rgba(255,255,255,.03)", borderRadius:8, padding:"5px 9px", border:"1px solid rgba(255,255,255,.06)" }}>
                {c.image && <Img src={c.image} alt={c.name} style={{ width:24,height:24,borderRadius:"50%",objectFit:"cover" }}/>}
                <div>
                  <div style={{ fontSize:11,fontWeight:600,color:"#e2e8f0" }}>{c.name}</div>
                  {c.cv && <div style={{ fontSize:10,color:"#64748b" }}>CV: {c.cv}</div>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Rekomendasi */}
      {card.recommendations?.length > 0 && (
        <div style={{ padding:"8px 14px", borderTop:"1px solid rgba(255,255,255,.05)" }}>
          <div style={{ fontSize:11,color:"#64748b",marginBottom:6,display:"flex",alignItems:"center",gap:5 }}>
            <Icon src={ICONS.recs} size={12}/> Rekomendasi Serupa
          </div>
          <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
            {card.recommendations.map((r,i)=>(
              <a key={i} href={`${BASE}/detail/${r.slug||r.title?.toLowerCase().replace(/\s+/g,"-")}`} target="_blank" rel="noreferrer"
                style={{ fontSize:11,padding:"3px 10px",borderRadius:8,background:"rgba(255,255,255,.04)",color:"#94a3b8",border:"1px solid rgba(255,255,255,.07)",textDecoration:"none",transition:"all .15s" }}
                onMouseEnter={e=>e.target.style.borderColor="rgba(139,92,246,.4)"}
                onMouseLeave={e=>e.target.style.borderColor="rgba(255,255,255,.07)"}
              >{r.title}</a>
            ))}
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div style={{ padding:"10px 14px", borderTop:"1px solid rgba(255,255,255,.05)", display:"flex", gap:8 }}>
        <a href={watchUrl} target="_blank" rel="noreferrer" style={{
          flex:1, display:"flex",alignItems:"center",justifyContent:"center",gap:7,
          background:"linear-gradient(135deg,#8b5cf6,#6366f1)", borderRadius:10,
          color:"white", fontSize:13, fontWeight:700, padding:"8px 0", textDecoration:"none",
          boxShadow:"0 4px 15px rgba(139,92,246,.35)", transition:"transform .15s,box-shadow .15s"
        }}
        onMouseEnter={e=>{e.currentTarget.style.transform="scale(1.02)";e.currentTarget.style.boxShadow="0 6px 20px rgba(139,92,246,.5)";}}
        onMouseLeave={e=>{e.currentTarget.style.transform="scale(1)";e.currentTarget.style.boxShadow="0 4px 15px rgba(139,92,246,.35)";}}
        ><Icon src={ICONS.play} size={15}/> Tonton</a>

        <a href={detailUrl} target="_blank" rel="noreferrer" style={{
          flex:1, display:"flex",alignItems:"center",justifyContent:"center",gap:7,
          background:"rgba(139,92,246,.12)", borderRadius:10, border:"1px solid rgba(139,92,246,.3)",
          color:"#a78bfa", fontSize:13, fontWeight:700, padding:"8px 0", textDecoration:"none", transition:"all .15s"
        }}
        onMouseEnter={e=>{e.currentTarget.style.background="rgba(139,92,246,.22)";}}
        onMouseLeave={e=>{e.currentTarget.style.background="rgba(139,92,246,.12)";}}
        ><Icon src={ICONS.info} size={15}/> Detail</a>
      </div>
    </div>
  );
}

// ── Anime Slider ──
function AnimeSlider({ items }) {
  const [idx, setIdx] = useState(0);
  if (!items?.length) return null;
  const visible = 3;
  const max = Math.max(0, items.length - visible);
  const prev = () => setIdx(i => Math.max(0, i-1));
  const next = () => setIdx(i => Math.min(max, i+1));
  const shown = items.slice(idx, idx + visible);

  return (
    <div style={{ margin:"12px 0" }}>
      <div style={{ display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:8 }}>
        <div style={{ fontSize:11,color:"#64748b",textTransform:"uppercase",letterSpacing:1.5 }}>Anime Ditemukan</div>
        <div style={{ display:"flex",gap:5 }}>
          <button onClick={prev} disabled={idx===0} style={{ width:26,height:26,borderRadius:8,background:"rgba(255,255,255,.05)",border:"1px solid rgba(255,255,255,.08)",cursor:idx===0?"not-allowed":"pointer",display:"flex",alignItems:"center",justifyContent:"center",opacity:idx===0?.35:1,transition:".15s" }}>
            <Icon src={ICONS.prev} size={13}/>
          </button>
          <button onClick={next} disabled={idx>=max} style={{ width:26,height:26,borderRadius:8,background:"rgba(255,255,255,.05)",border:"1px solid rgba(255,255,255,.08)",cursor:idx>=max?"not-allowed":"pointer",display:"flex",alignItems:"center",justifyContent:"center",opacity:idx>=max?.35:1,transition:".15s" }}>
            <Icon src={ICONS.next} size={13}/>
          </button>
        </div>
      </div>

      <div style={{ display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:8 }}>
        {shown.map((a,i)=>{
          const slug = a.slug || a.title?.toLowerCase().replace(/\s+/g,"-");
          return (
            <div key={i} style={{ background:"rgba(255,255,255,.03)",borderRadius:12,overflow:"hidden",border:"1px solid rgba(255,255,255,.06)",transition:"all .2s",cursor:"pointer" }}
              onMouseEnter={e=>{e.currentTarget.style.borderColor="rgba(139,92,246,.35)";e.currentTarget.style.transform="translateY(-2px)";}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor="rgba(255,255,255,.06)";e.currentTarget.style.transform="translateY(0)";}}
            >
              <div style={{ position:"relative",paddingBottom:"140%",overflow:"hidden" }}>
                {a.image
                  ? <Img src={a.image} alt={a.title} style={{ position:"absolute",inset:0,width:"100%",height:"100%",objectFit:"cover" }}/>
                  : <div style={{ position:"absolute",inset:0,background:"rgba(139,92,246,.1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,color:"#475569" }}>No Poster</div>
                }
                <div style={{ position:"absolute",inset:0,background:"linear-gradient(to top,rgba(7,7,15,.95) 0%,transparent 50%)" }}/>
                {a.rating && (
                  <div style={{ position:"absolute",top:6,right:6,display:"flex",alignItems:"center",gap:3,background:"rgba(0,0,0,.7)",backdropFilter:"blur(8px)",borderRadius:6,padding:"2px 7px",fontSize:11,fontWeight:700,color:"#fbbf24" }}>
                    <Icon src={ICONS.star} size={11}/> {a.rating}
                  </div>
                )}
                <div style={{ position:"absolute",bottom:0,left:0,right:0,padding:"8px" }}>
                  <div style={{ fontSize:11,fontWeight:700,color:"white",lineHeight:1.3,marginBottom:4 }}>{a.title}</div>
                  <div style={{ display:"flex",gap:4,flexWrap:"wrap" }}>
                    {a.status && <span style={{ fontSize:9,padding:"1px 6px",borderRadius:6,background:a.status==="Ongoing"?"rgba(52,211,153,.2)":"rgba(99,102,241,.2)",color:a.status==="Ongoing"?"#34d399":"#818cf8" }}>{a.status}</span>}
                    {a.episode && <span style={{ fontSize:9,padding:"1px 6px",borderRadius:6,background:"rgba(255,255,255,.1)",color:"#94a3b8" }}>Ep {a.episode}</span>}
                  </div>
                </div>
              </div>
              <div style={{ display:"flex",gap:5,padding:"7px" }}>
                <a href={`${BASE}/watch/${slug}`} target="_blank" rel="noreferrer"
                  style={{ flex:1,display:"flex",alignItems:"center",justifyContent:"center",gap:4,background:"rgba(139,92,246,.8)",borderRadius:8,color:"white",fontSize:10,fontWeight:700,padding:"5px 0",textDecoration:"none",transition:".15s" }}
                  onMouseEnter={e=>e.currentTarget.style.background="rgba(139,92,246,1)"}
                  onMouseLeave={e=>e.currentTarget.style.background="rgba(139,92,246,.8)"}
                ><Icon src={ICONS.play} size={11}/> Watch</a>
                <a href={`${BASE}/detail/${slug}`} target="_blank" rel="noreferrer"
                  style={{ flex:1,display:"flex",alignItems:"center",justifyContent:"center",gap:4,background:"rgba(255,255,255,.05)",borderRadius:8,color:"#94a3b8",fontSize:10,fontWeight:700,padding:"5px 0",textDecoration:"none",border:"1px solid rgba(255,255,255,.08)",transition:".15s" }}
                  onMouseEnter={e=>e.currentTarget.style.borderColor="rgba(139,92,246,.4)"}
                  onMouseLeave={e=>e.currentTarget.style.borderColor="rgba(255,255,255,.08)"}
                ><Icon src={ICONS.info} size={11}/> Detail</a>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Table renderer ──
function TableRenderer({ text }) {
  const tableRx = /(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)/g;
  const parts = []; let last = 0, m;
  while ((m = tableRx.exec(text)) !== null) {
    if (m.index > last) parts.push({ type:"text", content: text.slice(last, m.index) });
    parts.push({ type:"table", content: m[0] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ type:"text", content: text.slice(last) });
  if (!parts.length) parts.push({ type:"text", content: text });

  return (
    <>{parts.map((p,i) => p.type === "table"
      ? <RenderTable key={i} raw={p.content}/>
      : <RenderText key={i} text={p.content}/>
    )}</>
  );
}

function RenderTable({ raw }) {
  const lines = raw.trim().split("\n").filter(l=>l.trim()&&l.includes("|"));
  if (lines.length < 2) return <RenderText text={raw}/>;
  const headers = lines[0].split("|").map(h=>h.trim()).filter(Boolean);
  const body = lines.slice(2).map(l=>l.split("|").map(c=>c.trim()).filter(Boolean));
  return (
    <div style={{ overflowX:"auto",margin:"10px 0",borderRadius:12,border:"1px solid rgba(139,92,246,.2)",overflow:"hidden" }}>
      <table style={{ width:"100%",borderCollapse:"collapse",fontSize:12 }}>
        <thead><tr>{headers.map((h,i)=>
          <th key={i} style={{ padding:"8px 14px",textAlign:"left",background:"rgba(139,92,246,.18)",color:"#c4b5fd",fontWeight:700,fontSize:11,textTransform:"uppercase",letterSpacing:.8,whiteSpace:"nowrap" }}>{h}</th>
        )}</tr></thead>
        <tbody>{body.map((row,i)=>
          <tr key={i} style={{ borderTop:"1px solid rgba(255,255,255,.05)",background:i%2===0?"rgba(255,255,255,.015)":"transparent" }}>
            {row.map((cell,j)=><td key={j} style={{ padding:"8px 14px",color:"#e2e8f0",fontSize:12 }}>{cell}</td>)}
          </tr>
        )}</tbody>
      </table>
    </div>
  );
}

function RenderText({ text }) {
  const renderInline = t => {
    const parts = t.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    return parts.map((p,i)=>{
      if (p.startsWith("**")&&p.endsWith("**")) return <strong key={i} style={{color:"#a78bfa",fontWeight:700}}>{p.slice(2,-2)}</strong>;
      if (p.startsWith("`")&&p.endsWith("`"))   return <code key={i} style={{background:"rgba(139,92,246,.15)",color:"#c4b5fd",padding:"1px 5px",borderRadius:4,fontSize:12,fontFamily:"'JetBrains Mono',monospace"}}>{p.slice(1,-1)}</code>;
      return <span key={i}>{p}</span>;
    });
  };
  return (
    <div>{text.split("\n").map((line,i)=>{
      const key=`r${i}`;
      if(!line.trim()) return <div key={key} style={{height:5}}/>;
      if(/^#{1,2}\s/.test(line)) return <div key={key} style={{fontSize:14,fontWeight:800,color:"#c4b5fd",margin:"10px 0 4px"}}>{line.replace(/^#+\s/,"")}</div>;
      if(/^[-•*]\s/.test(line)) return <div key={key} style={{display:"flex",gap:7,margin:"3px 0",alignItems:"flex-start"}}><span style={{color:"#8b5cf6",fontSize:8,marginTop:5,flexShrink:0}}>◆</span><span style={{lineHeight:1.7}}>{renderInline(line.replace(/^[-•*]\s/,""))}</span></div>;
      if(/^\d+\.\s/.test(line)) return <div key={key} style={{display:"flex",gap:7,margin:"3px 0",alignItems:"flex-start"}}><span style={{color:"#8b5cf6",minWidth:18,flexShrink:0,fontWeight:700,fontSize:13}}>{line.match(/^\d+/)[0]}.</span><span style={{lineHeight:1.7}}>{renderInline(line.replace(/^\d+\.\s/,""))}</span></div>;
      return <p key={key} style={{margin:"3px 0",lineHeight:1.75}}>{renderInline(line)}</p>;
    })}</div>
  );
}

// ── Message bubble ──
function Message({ msg }) {
  if (msg.role === "user") return (
    <div style={{ display:"flex",justifyContent:"flex-end" }}>
      <div style={{ maxWidth:"76%",display:"flex",flexDirection:"column",gap:3,alignItems:"flex-end" }}>
        <div style={{ padding:"11px 16px",borderRadius:"18px 18px 4px 18px",background:"linear-gradient(135deg,#8b5cf6,#6366f1)",color:"white",fontSize:14,lineHeight:1.65,boxShadow:"0 4px 20px rgba(139,92,246,.3)" }}>
          {msg.content}
        </div>
        <div style={{ fontSize:10,color:"#334155",fontFamily:"'JetBrains Mono',monospace" }}>{fmt(msg.time)}</div>
      </div>
    </div>
  );

  return (
    <div style={{ display:"flex",gap:10,alignItems:"flex-start" }}>
      <div style={{ width:32,height:32,borderRadius:10,flexShrink:0,background:"linear-gradient(135deg,#8b5cf6,#6366f1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:15,boxShadow:"0 0 12px rgba(139,92,246,.3)",marginTop:2 }}>🎌</div>
      <div style={{ maxWidth:"82%",display:"flex",flexDirection:"column",gap:6 }}>
        <div style={{ padding:"12px 16px",borderRadius:"4px 18px 18px 18px",background:msg.err?"rgba(239,68,68,.08)":"rgba(255,255,255,.045)",border:msg.err?"1px solid rgba(239,68,68,.2)":"1px solid rgba(139,92,246,.12)",fontSize:14,color:"#e2e8f0" }}>
          <TableRenderer text={msg.content}/>
          <AnimeCard card={msg.animeCard}/>
          <AnimeSlider items={msg.animeSlider}/>
        </div>
        {msg.sources?.length>0 && (
          <div style={{ display:"flex",flexWrap:"wrap",gap:4,paddingLeft:4 }}>
            {msg.sources.map(s=>(
              <span key={s} style={{ fontSize:10,padding:"2px 8px",borderRadius:20,background:"rgba(139,92,246,.08)",color:"#7c5ccc",border:"1px solid rgba(139,92,246,.15)" }}>📺 {s}</span>
            ))}
          </div>
        )}
        <div style={{ fontSize:10,color:"#334155",paddingLeft:4,fontFamily:"'JetBrains Mono',monospace" }}>{fmt(msg.time)}{msg.rt&&` · ${msg.rt}s`}</div>
      </div>
    </div>
  );
}

function Dots() {
  return <div style={{display:"flex",gap:5}}>{[0,1,2].map(i=><div key={i} style={{width:7,height:7,borderRadius:"50%",background:"linear-gradient(135deg,#8b5cf6,#6366f1)",animation:"bounce 1.3s infinite",animationDelay:`${i*.18}s`}}/>)}</div>;
}

const fmt = d => d.toLocaleTimeString("id-ID",{hour:"2-digit",minute:"2-digit"});

// ── MAIN APP ──
export default function App() {
  const [msgs, setMsgs] = useState([{
    role:"assistant",
    content:"Yo! Gue **AniBot** 🎌\nTemen ngobrol lo yang jago soal anime & topik umum.\nMau nanya apa bro?",
    time:new Date(), sources:[], animeCard:null, animeSlider:[],
  }]);
  const [input,setInput]     = useState("");
  const [loading,setLoading] = useState(false);
  const [stats,setStats]     = useState(null);
  const [sidebar,setSidebar] = useState(false);
  const endRef   = useRef(null);
  const inputRef = useRef(null);

  useEffect(()=>{ endRef.current?.scrollIntoView({behavior:"smooth"}); },[msgs,loading]);
  useEffect(()=>{ fetch(`${API}/stats`).then(r=>r.json()).then(setStats).catch(()=>null); },[]);

  const send = useCallback(async (text) => {
    const msg = (text||input).trim();
    if (!msg||loading) return;
    setInput("");
    if (inputRef.current) { inputRef.current.style.height="auto"; }
    setLoading(true);
    const userMsg = {role:"user",content:msg,time:new Date()};
    const allMsgs = [...msgs,userMsg];
    setMsgs(allMsgs);
    const history = allMsgs.slice(1).map(m=>({role:m.role,content:m.content}));
    try {
      const r = await fetch(`${API}/chat`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg,history})});
      if (!r.ok){const e=await r.json();throw new Error(e.detail||"Error");}
      const d = await r.json();
      setMsgs(p=>[...p,{role:"assistant",content:d.reply,time:new Date(),sources:d.sources||[],animeCard:d.anime_card||null,animeSlider:d.anime_slider||[],rt:d.response_time}]);
    } catch(e) {
      setMsgs(p=>[...p,{role:"assistant",content:`Waduh error bro 😅\n${e.message}`,time:new Date(),sources:[],animeCard:null,animeSlider:[],err:true}]);
    } finally {
      setLoading(false);
      setTimeout(()=>inputRef.current?.focus(),100);
    }
  }, [msgs, input, loading]);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
        *,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
        html,body{height:100%;background:#07070f;color:#e2e8f0;font-family:'Outfit',sans-serif;overflow:hidden;}
        @keyframes bounce{0%,80%,100%{transform:scale(.5);opacity:.4}40%{transform:scale(1);opacity:1}}
        @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:rgba(139,92,246,.3);border-radius:2px}
        .msg{animation:fadeUp .22s cubic-bezier(.22,1,.36,1);}
        .send-btn{background:linear-gradient(135deg,#8b5cf6,#6366f1);border:none;border-radius:12px;width:40px;height:40px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:transform .15s,box-shadow .15s;flex-shrink:0;box-shadow:0 0 16px rgba(139,92,246,.3);}
        .send-btn:hover{transform:scale(1.08);box-shadow:0 0 28px rgba(139,92,246,.55);}
        .send-btn:active{transform:scale(.93);}
        .send-btn:disabled{opacity:.3;cursor:not-allowed;transform:none;box-shadow:none;}
        .chip{background:rgba(255,255,255,.04);border:1px solid rgba(139,92,246,.2);border-radius:20px;color:#94a3b8;font-size:12px;padding:6px 14px;cursor:pointer;font-family:'Outfit',sans-serif;transition:all .18s;}
        .chip:hover{background:rgba(139,92,246,.14);border-color:rgba(139,92,246,.45);color:#e2e8f0;}
        .inp-box{background:rgba(255,255,255,.04);border:1.5px solid rgba(139,92,246,.18);border-radius:16px;transition:border-color .2s,box-shadow .2s;}
        .inp-box:focus-within{border-color:rgba(139,92,246,.55);box-shadow:0 0 0 4px rgba(139,92,246,.07);}
        .sq-btn{padding:7px 12px;border-radius:10px;font-size:12px;color:#94a3b8;cursor:pointer;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);transition:all .15s;margin-bottom:4px;font-family:'Outfit',sans-serif;text-align:left;width:100%;}
        .sq-btn:hover{background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.28);color:#e2e8f0;}
      `}</style>

      {/* BG */}
      <div style={{position:"fixed",inset:0,zIndex:0,pointerEvents:"none",backgroundImage:"linear-gradient(rgba(139,92,246,.022) 1px,transparent 1px),linear-gradient(90deg,rgba(139,92,246,.022) 1px,transparent 1px)",backgroundSize:"44px 44px"}}/>
      <div style={{position:"fixed",top:-200,right:-150,width:500,height:500,borderRadius:"50%",background:"radial-gradient(circle,rgba(139,92,246,.07) 0%,transparent 70%)",pointerEvents:"none",zIndex:0}}/>
      <div style={{position:"fixed",bottom:-180,left:-150,width:400,height:400,borderRadius:"50%",background:"radial-gradient(circle,rgba(99,102,241,.05) 0%,transparent 70%)",pointerEvents:"none",zIndex:0}}/>
      <div style={{position:"fixed",top:0,left:0,right:0,height:"1px",background:"linear-gradient(90deg,transparent,rgba(139,92,246,.4),transparent)",zIndex:1}}/>

      <div style={{display:"flex",height:"100vh",overflow:"hidden",position:"relative",zIndex:1}}>

        {/* SIDEBAR */}
        <div style={{width:sidebar?270:0,overflow:"hidden",transition:"width .3s cubic-bezier(.4,0,.2,1)",background:"rgba(7,7,15,.97)",borderRight:sidebar?"1px solid rgba(139,92,246,.18)":"none",flexShrink:0,backdropFilter:"blur(20px)"}}>
          <div style={{width:270,padding:"18px 14px",height:"100%",overflowY:"auto",display:"flex",flexDirection:"column",gap:16}}>
            <div style={{paddingBottom:14,borderBottom:"1px solid rgba(255,255,255,.05)"}}>
              <div style={{fontSize:20,fontWeight:900,letterSpacing:-1}}>Ani<span style={{background:"linear-gradient(135deg,#8b5cf6,#6366f1)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>Bot</span></div>
              <div style={{fontSize:11,color:"#334155",marginTop:2}}>AI Anime · Ollama · v3.0</div>
            </div>

            {stats ? (
              <div>
                <div style={{fontSize:10,color:"#334155",textTransform:"uppercase",letterSpacing:2,marginBottom:8}}>Dataset</div>
                {[["Detail",stats.detail_count,"#8b5cf6"],["List",stats.listing_count,"#6366f1"],["Total",stats.total,"#34d399"],["Studios",stats.studios,"#f59e0b"]].map(([l,v,c])=>(
                  <div key={l} style={{background:"rgba(255,255,255,.03)",border:"1px solid rgba(255,255,255,.05)",borderRadius:10,padding:"8px 12px",display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:5}}>
                    <span style={{fontSize:12,color:"#64748b"}}>{l}</span>
                    <span style={{fontSize:17,fontWeight:800,color:c,fontFamily:"'JetBrains Mono',monospace"}}>{v?.toLocaleString()||0}</span>
                  </div>
                ))}
                <div style={{marginTop:8}}>
                  <div style={{fontSize:10,color:"#334155",textTransform:"uppercase",letterSpacing:2,marginBottom:6}}>Top Genre</div>
                  {(stats.top_genres||[]).slice(0,6).map(g=>(
                    <div key={g.genre} style={{display:"flex",justifyContent:"space-between",fontSize:12,padding:"4px 0",borderBottom:"1px solid rgba(255,255,255,.04)"}}>
                      <span style={{color:"#64748b"}}>{g.genre}</span>
                      <span style={{color:"#8b5cf6",fontFamily:"'JetBrains Mono',monospace",fontWeight:600}}>{g.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : <div style={{fontSize:12,color:"#334155"}}>Backend offline</div>}

            <div>
              <div style={{fontSize:10,color:"#334155",textTransform:"uppercase",letterSpacing:2,marginBottom:8}}>Contoh Tanya</div>
              {SUGGESTIONS.map(s=><button key={s} className="sq-btn" onClick={()=>send(s)}>{s}</button>)}
            </div>
          </div>
        </div>

        {/* MAIN */}
        <div style={{flex:1,display:"flex",flexDirection:"column",minWidth:0}}>
          {/* Header */}
          <div style={{padding:"11px 16px",display:"flex",alignItems:"center",gap:12,background:"rgba(7,7,15,.92)",backdropFilter:"blur(20px)",borderBottom:"1px solid rgba(139,92,246,.12)"}}>
            <button onClick={()=>setSidebar(!sidebar)} style={{background:"none",border:"none",cursor:"pointer",padding:4,borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center"}}>
              <Icon src={sidebar?ICONS.close:ICONS.menu} size={20}/>
            </button>
            <div style={{width:36,height:36,borderRadius:10,flexShrink:0,background:"linear-gradient(135deg,#8b5cf6,#6366f1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:17,boxShadow:"0 0 16px rgba(139,92,246,.35)"}}>🎌</div>
            <div style={{flex:1}}>
              <div style={{fontWeight:800,fontSize:16,letterSpacing:-.5}}>Ani<span style={{background:"linear-gradient(135deg,#8b5cf6,#6366f1)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>Bot</span></div>
              <div style={{fontSize:11,color:"#334155"}}>{stats?`${stats.total?.toLocaleString()} anime · ${stats.model}`:"Connecting..."}</div>
            </div>
            <div style={{display:"flex",alignItems:"center",gap:6}}>
              <div style={{width:7,height:7,borderRadius:"50%",background:stats?"#34d399":"#ef4444",boxShadow:stats?"0 0 8px #34d399":"none"}}/>
              <span style={{fontSize:11,color:"#334155"}}>{stats?"Online":"Offline"}</span>
            </div>
            <button onClick={()=>setMsgs([{role:"assistant",content:"Reset! Mau nanya apa bro? 🎌",time:new Date(),sources:[],animeCard:null,animeSlider:[]}])} style={{background:"rgba(255,255,255,.04)",border:"1px solid rgba(255,255,255,.07)",color:"#64748b",cursor:"pointer",borderRadius:10,padding:"5px 12px",fontSize:12,fontFamily:"'Outfit',sans-serif",display:"flex",alignItems:"center",gap:5,transition:".15s"}}
              onMouseEnter={e=>e.currentTarget.style.borderColor="rgba(139,92,246,.35)"}
              onMouseLeave={e=>e.currentTarget.style.borderColor="rgba(255,255,255,.07)"}
            ><Icon src={ICONS.reset} size={13}/> Reset</button>
          </div>

          {/* Messages */}
          <div style={{flex:1,overflowY:"auto",padding:"18px 14px",display:"flex",flexDirection:"column",gap:14}}>
            {msgs.map((m,i)=><div key={i} className="msg"><Message msg={m}/></div>)}
            {loading && (
              <div className="msg" style={{display:"flex",gap:10,alignItems:"flex-start"}}>
                <div style={{width:32,height:32,borderRadius:10,flexShrink:0,background:"linear-gradient(135deg,#8b5cf6,#6366f1)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:15}}>🎌</div>
                <div style={{padding:"12px 16px",background:"rgba(255,255,255,.04)",border:"1px solid rgba(139,92,246,.12)",borderRadius:"4px 18px 18px 18px"}}><Dots/></div>
              </div>
            )}
            <div ref={endRef}/>
          </div>

          {/* Suggestions awal */}
          {msgs.length<=1&&!loading&&(
            <div style={{padding:"0 14px 10px",display:"flex",flexWrap:"wrap",gap:6}}>
              {SUGGESTIONS.map(s=><button key={s} className="chip" onClick={()=>send(s)}>{s}</button>)}
            </div>
          )}

          {/* Input */}
          <div style={{padding:"10px 14px 14px",borderTop:"1px solid rgba(139,92,246,.1)",background:"rgba(7,7,15,.92)",backdropFilter:"blur(20px)"}}>
            <div className="inp-box" style={{display:"flex",alignItems:"flex-end",gap:8,padding:"8px 8px 8px 15px"}}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={e=>setInput(e.target.value)}
                onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}}
                placeholder="Tanya apa aja bro... anime, rekomendasi, atau hal umum 🎌"
                rows={1}
                disabled={loading}
                style={{flex:1,background:"none",border:"none",outline:"none",color:"#e2e8f0",fontSize:14,resize:"none",fontFamily:"'Outfit',sans-serif",lineHeight:1.6,maxHeight:120,overflowY:"auto"}}
                onInput={e=>{e.target.style.height="auto";e.target.style.height=e.target.scrollHeight+"px";}}
              />
              <button className="send-btn" onClick={()=>send()} disabled={loading||!input.trim()}>
                {loading ? "⏳" : <Icon src={ICONS.send} size={18}/>}
              </button>
            </div>
            <div style={{textAlign:"center",fontSize:10,color:"#1e293b",marginTop:7,fontFamily:"'JetBrains Mono',monospace"}}>
              Enter kirim · Shift+Enter baris baru · AniBot v3.0
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
JSEOF
ok "frontend/src/App.jsx ditulis ulang"

# ─────────────────────────────────────────
# 4. INSTALL DEPS & BUILD
# ─────────────────────────────────────────
step "Install & Build Frontend"
cd "$FRONT"
npm install --silent
ok "npm install selesai"

# ─────────────────────────────────────────
# 5. RESTART SERVICES
# ─────────────────────────────────────────
step "Restart Services"
pm2 restart anibot-backend  --update-env
pm2 restart anibot-frontend --update-env
sleep 4
pm2 status

# ─────────────────────────────────────────
# 6. TEST
# ─────────────────────────────────────────
step "Test API"
echo ""
time curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"jelasin anime medalist dong","history":[]}' \
  | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('Reply:',d['reply'][:120])
print('Card :', 'ADA ✓' if d.get('anime_card') else 'kosong')
print('Slider:', len(d.get('anime_slider',[])), 'item')
print('Waktu:', d['response_time'],'detik')
"

echo ""
printf "${G}${W}✓ Upgrade selesai!${N}\n"
printf "${C}  Frontend : http://137.184.60.192:5173${N}\n"
printf "${C}  Backend  : http://137.184.60.192:8000${N}\n"
printf "${C}  API Docs : http://137.184.60.192:8000/docs${N}\n\n"
