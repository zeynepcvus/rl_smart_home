const pct = (rl, rb) => rb === 0 ? 0 : ((rb - rl) / Math.abs(rb)) * 100;
const fmtPct = (p) => Math.abs(p).toFixed(1);

export default function Comparison({ goTo, formData, apiResult }) {
  const rlSum = apiResult?.rl?.summary || {};
  const rbSum = apiResult?.rule_based?.summary || {};

  const RL = {
    cost: rlSum.total_cost ?? 0,
    comfort: rlSum.comfort_violations ?? 0,
    deadline: rlSum.deadline_violations ?? 0,
    hvac: rlSum.hvac_switches ?? 0,
    invalid: rlSum.invalid_actions ?? 0,
  };
  const RB = {
    cost: rbSum.total_cost ?? 0,
    comfort: rbSum.comfort_violations ?? 0,
    deadline: rbSum.deadline_violations ?? 0,
    hvac: rbSum.hvac_switches ?? 0,
    invalid: rbSum.invalid_actions ?? 0,
  };

  const MODE_LABELS = {
    cost: "Maliyet Odaklı",
    balanced: "Dengeli",
    comfort: "Konfor Odaklı",
  };
  const modeName = MODE_LABELS[formData?.mode] || "Dengeli";

  const rlWins = RL.cost < RB.cost;

  const maxVal = (a, b) => Math.max(a, b, 0.01);

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", flexDirection: "column", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{ position: "fixed", inset: 0, backgroundImage: "linear-gradient(rgba(29,158,117,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.04) 1px, transparent 1px)", backgroundSize: "40px 40px", pointerEvents: "none" }} />

      <div style={{ display: "flex", alignItems: "center", padding: ".85rem 1.5rem", borderBottom: "0.5px solid rgba(255,255,255,0.07)", position: "relative", zIndex: 1, gap: "1rem" }}>
        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5", display: "flex", alignItems: "center", gap: 6, marginRight: "auto" }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#1D9E75", animation: "blink 2s ease infinite" }} />
          SmartHome RL
        </div>
        <span onClick={() => goTo("dashboard")} style={{ fontSize: 12, color: "rgba(240,244,248,0.4)", padding: "5px 12px", borderRadius: 6, cursor: "pointer" }}>Dashboard</span>
        <span style={{ fontSize: 12, color: "#5DCAA5", padding: "5px 12px", borderRadius: 6, background: "rgba(29,158,117,0.12)", fontWeight: 500 }}>Karşılaştırma</span>
        <span onClick={() => goTo("welcome")} style={{ fontSize: 12, color: "rgba(240,244,248,0.25)", padding: "5px 12px", borderRadius: 6, cursor: "pointer", borderLeft: "0.5px solid rgba(255,255,255,0.07)", marginLeft: 4 }}>Yeniden Başla</span>
      </div>

      <div style={{ flex: 1, padding: "1.25rem 1.5rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 12 }}>

        {!apiResult && (
          <div style={{ textAlign: "center", padding: "3rem", color: "rgba(240,244,248,0.35)", fontSize: 14 }}>
            Simülasyon verisi yok. Lütfen önce simülasyonu başlatın.
          </div>
        )}

        {apiResult && <>
          {/* Hero */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <div style={{ background: "rgba(29,158,117,0.08)", border: "1.5px solid #1D9E75", borderRadius: 10, padding: "1rem", textAlign: "center" }}>
              <div style={{ fontSize: 10, padding: "2px 10px", borderRadius: 20, background: "rgba(29,158,117,0.2)", color: "#5DCAA5", fontWeight: 500, display: "inline-block", marginBottom: 6 }}>RL Ajanı · PPO</div>
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.6)", marginBottom: 6 }}>{modeName} Modu</div>
              <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 30, color: "#5DCAA5" }}>{RL.cost} TL</div>
              <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginTop: 2 }}>günlük maliyet</div>
              {rlWins && (
                <div style={{ display: "inline-flex", alignItems: "center", gap: 4, background: "rgba(29,158,117,0.15)", borderRadius: 20, padding: "3px 10px", fontSize: 10, color: "#5DCAA5", fontWeight: 500, marginTop: 6 }}>
                  🏆 Daha iyi performans
                </div>
              )}
            </div>

            <div style={{ background: "rgba(255,255,255,0.02)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: "1rem", display: "flex", flexDirection: "column", justifyContent: "center" }}>
              <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".5rem" }}>Değerlendirme yöntemi</div>
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.5)", lineHeight: 1.6 }}>
                Aynı senaryo üzerinde <span style={{ color: "rgba(240,244,248,0.7)", fontWeight: 500 }}>paired evaluation</span> — her iki ajan da aynı günü yaşadı.
              </div>
            </div>

            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: "1rem", textAlign: "center" }}>
              <div style={{ fontSize: 10, padding: "2px 10px", borderRadius: 20, background: "rgba(255,255,255,0.07)", color: "rgba(240,244,248,0.45)", fontWeight: 500, display: "inline-block", marginBottom: 6 }}>Kural Tabanlı</div>
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.6)", marginBottom: 6 }}>Rule-Based Agent</div>
              <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 30, color: "#f0f4f8" }}>{RB.cost} TL</div>
              <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginTop: 2 }}>günlük maliyet</div>
              <div style={{ fontSize: 10, color: "rgba(240,244,248,0.25)", marginTop: 6 }}>Deterministik · öğrenmiyor</div>
            </div>
          </div>

          {/* Metrik tablosu */}
          <div style={{ background: "rgba(255,255,255,0.02)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: ".85rem 1rem" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr 1fr", gap: 8, paddingBottom: 6, borderBottom: "0.5px solid rgba(255,255,255,0.07)", marginBottom: 4 }}>
              {["Metrik", "RL Ajanı", "Kural Tabanlı"].map(h => (
                <div key={h} style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.3)", letterSpacing: ".05em", textTransform: "uppercase", textAlign: h !== "Metrik" ? "center" : "left" }}>{h}</div>
              ))}
            </div>
            {[
              { label: "Günlük maliyet (TL)", rl: RL.cost, rb: RB.cost, showPct: true },
              { label: "Konfor ihlali", rl: RL.comfort, rb: RB.comfort, showPct: true },
              { label: "Deadline ihlali", rl: RL.deadline, rb: RB.deadline, showPct: false },
              { label: "HVAC anahtar sayısı", rl: RL.hvac, rb: RB.hvac, showPct: true },
              { label: "Geçersiz aksiyon", rl: RL.invalid, rb: RB.invalid, showPct: false },
            ].map(row => {
              const same = row.rl === row.rb;
              return (
                <div key={row.label} style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr 1fr", gap: 8, padding: "7px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)", alignItems: "center" }}>
                  <span style={{ fontSize: 12, color: "rgba(240,244,248,0.55)" }}>{row.label}</span>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
                    <span style={{ fontSize: 14, fontWeight: 500, color: same ? "rgba(240,244,248,0.4)" : "#5DCAA5" }}>{row.rl}</span>
                    {row.showPct && !same && (() => {
                      const p = pct(row.rl, row.rb);
                      const better = p > 0;
                      return (
                        <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: better ? "rgba(29,158,117,0.15)" : "rgba(226,75,74,0.15)", color: better ? "#5DCAA5" : "#e24b4a" }}>
                          {better ? `-${fmtPct(p)}%` : `+${fmtPct(p)}%`}
                        </span>
                      );
                    })()}
                  </div>
                  <span style={{ fontSize: 14, fontWeight: 500, color: same ? "rgba(240,244,248,0.4)" : "rgba(240,244,248,0.6)", textAlign: "center" }}>{row.rb}</span>
                </div>
              );
            })}
          </div>

          {/* Alt satır */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div style={{ background: "rgba(255,255,255,0.02)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase" }}>Görsel karşılaştırma</div>
                <div style={{ display: "flex", gap: 10 }}>
                  {[{ color: "#1D9E75", label: "RL" }, { color: "rgba(240,244,248,0.25)", label: "Kural" }].map(l => (
                    <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: l.color }} />
                      <span style={{ fontSize: 10, color: "rgba(240,244,248,0.35)" }}>{l.label}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "flex-end", height: 140 }}>
                {[
                  { label: "Maliyet (TL)", rlV: RL.cost, rbV: RB.cost },
                  { label: "Konfor ihlali", rlV: RL.comfort, rbV: RB.comfort },
                  { label: "HVAC anahtarı", rlV: RL.hvac, rbV: RB.hvac },
                ].map(b => {
                  const mx = maxVal(b.rlV, b.rbV);
                  const rlH = Math.round((b.rlV / mx) * 90);
                  const rbH = Math.round((b.rbV / mx) * 90);
                  const rlBetter = b.rlV <= b.rbV;
                  return (
                    <div key={b.label} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>
                      <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 110 }}>
                        {[
                          { h: rlH, val: b.rlV, color: "#1D9E75", textColor: "#5DCAA5", better: rlBetter },
                          { h: rbH, val: b.rbV, color: "rgba(240,244,248,0.18)", textColor: "rgba(240,244,248,0.4)", better: !rlBetter },
                        ].map((bar, bi) => (
                          <div key={bi} style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", height: "100%" }}>
                            <span style={{ fontSize: 10, color: bar.textColor, fontWeight: 500, marginBottom: 3 }}>{bar.val}</span>
                            <div style={{
                              width: 22, height: bar.h,
                              background: bar.color,
                              borderRadius: "3px 3px 0 0",
                              transition: "height .4s ease",
                              boxShadow: bar.better ? `0 0 8px ${bar.color}88` : "none"
                            }} />
                          </div>
                        ))}
                      </div>
                      <div style={{ width: "100%", height: 1, background: "rgba(255,255,255,0.08)" }} />
                      <span style={{ fontSize: 9, color: "rgba(240,244,248,0.3)", textAlign: "center", marginTop: 5, lineHeight: 1.3 }}>{b.label}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ background: "rgba(255,255,255,0.02)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Öne çıkan bulgular</div>
              {[
                { text: RL.cost <= RB.cost
                    ? <span>RL ajanı <strong style={{ color: "#5DCAA5" }}>%{fmtPct(pct(RL.cost, RB.cost))} daha ucuz</strong> enerji kullandı.</span>
                    : <span>RL ajanı kural tabanlıdan <strong style={{ color: "#e24b4a" }}>%{fmtPct(pct(RL.cost, RB.cost))} daha pahalı</strong> çalıştı.</span>,
                  green: RL.cost <= RB.cost },
                { text: RL.comfort <= RB.comfort
                    ? <span>Konfor ihlali <strong style={{ color: "#5DCAA5" }}>%{fmtPct(pct(RL.comfort, RB.comfort))} azaldı</strong> — HVAC daha stabil çalıştı.</span>
                    : <span>Konfor ihlali <strong style={{ color: "#e24b4a" }}>%{fmtPct(pct(RL.comfort, RB.comfort))} arttı</strong> — kural tabanlı daha stabil çalıştı.</span>,
                  green: RL.comfort <= RB.comfort },
                { text: <span>Deadline ihlali: RL <strong style={{ color: "#5DCAA5" }}>{RL.deadline}</strong>, Kural tabanlı <strong style={{ color: "#5DCAA5" }}>{RB.deadline}</strong>.</span>, green: RL.deadline <= RB.deadline },
                { text: <span>Sonuçlar <strong style={{ color: "#FAC775" }}>simülasyon ortamında</strong> elde edildi.</span>, green: false },
              ].map((b, i) => (
                <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "5px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: b.green ? "#1D9E75" : "#FAC775", marginTop: 5, flexShrink: 0 }} />
                  <div style={{ fontSize: 12, color: "rgba(240,244,248,0.5)", lineHeight: 1.5 }}>{b.text}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ textAlign: "center", fontSize: 11, color: "rgba(240,244,248,0.25)" }}>
            Aynı senaryo üzerinde paired evaluation · seed=42
          </div>
        </>}
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
