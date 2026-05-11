const RL = {
  cost: 33.16, comfort: 2.89, deadline: 0.00,
  hvac: 5.72, invalid: 0.00, reward: -12.67
};
const RB = {
  cost: 34.78, comfort: 5.80, deadline: 0.00,
  hvac: 8.89, invalid: 0.00, reward: -17.13
};

const pct = (rl, rb) => (((rb - rl) / rb) * 100).toFixed(1);

export default function Comparison({ goTo }) {
  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", flexDirection: "column", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.04) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      {/* Topbar */}
      <div style={{
        display: "flex", alignItems: "center", padding: ".85rem 1.5rem",
        borderBottom: "0.5px solid rgba(255,255,255,0.07)", position: "relative", zIndex: 1, gap: "1rem"
      }}>
        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5", display: "flex", alignItems: "center", gap: 6, marginRight: "auto" }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#1D9E75", animation: "blink 2s ease infinite" }} />
          SmartHome RL
        </div>
        <span
          onClick={() => goTo("dashboard")}
          style={{ fontSize: 12, color: "rgba(240,244,248,0.4)", padding: "5px 12px", borderRadius: 6, cursor: "pointer" }}
        >Dashboard</span>
        <span style={{
          fontSize: 12, color: "#5DCAA5", padding: "5px 12px", borderRadius: 6,
          background: "rgba(29,158,117,0.12)", fontWeight: 500
        }}>Karşılaştırma</span>
      </div>

      <div style={{ flex: 1, padding: "1.25rem 1.5rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 12 }}>

        {/* Hero */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
          <div style={{
            background: "rgba(29,158,117,0.08)", border: "1.5px solid #1D9E75",
            borderRadius: 10, padding: "1rem", textAlign: "center"
          }}>
            <div style={{ fontSize: 10, padding: "2px 10px", borderRadius: 20, background: "rgba(29,158,117,0.2)", color: "#5DCAA5", fontWeight: 500, display: "inline-block", marginBottom: 6 }}>RL Ajanı · PPO</div>
            <div style={{ fontSize: 12, color: "rgba(240,244,248,0.6)", marginBottom: 6 }}>Balanced Model v2</div>
            <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 30, color: "#5DCAA5" }}>{RL.cost} TL</div>
            <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginTop: 2 }}>ort. günlük maliyet</div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: 4, background: "rgba(29,158,117,0.15)", borderRadius: 20, padding: "3px 10px", fontSize: 10, color: "#5DCAA5", fontWeight: 500, marginTop: 6 }}>
              🏆 Daha iyi performans
            </div>
          </div>

          <div style={{
            background: "rgba(255,255,255,0.02)", border: "0.5px solid rgba(255,255,255,0.07)",
            borderRadius: 10, padding: "1rem", display: "flex", flexDirection: "column", justifyContent: "center"
          }}>
            <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".5rem" }}>Değerlendirme yöntemi</div>
            <div style={{ fontSize: 12, color: "rgba(240,244,248,0.5)", lineHeight: 1.6 }}>
              100 özdeş senaryo üzerinde <span style={{ color: "rgba(240,244,248,0.7)", fontWeight: 500 }}>paired evaluation</span> — her iki ajan da aynı günü yaşadı.
            </div>
          </div>

          <div style={{
            background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.1)",
            borderRadius: 10, padding: "1rem", textAlign: "center"
          }}>
            <div style={{ fontSize: 10, padding: "2px 10px", borderRadius: 20, background: "rgba(255,255,255,0.07)", color: "rgba(240,244,248,0.45)", fontWeight: 500, display: "inline-block", marginBottom: 6 }}>Kural Tabanlı</div>
            <div style={{ fontSize: 12, color: "rgba(240,244,248,0.6)", marginBottom: 6 }}>Rule-Based Agent</div>
            <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 30, color: "#f0f4f8" }}>{RB.cost} TL</div>
            <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginTop: 2 }}>ort. günlük maliyet</div>
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
            { label: "Günlük maliyet (TL)", rl: RL.cost, rb: RB.cost, better: true, suffix: "", showPct: true },
            { label: "Konfor ihlali", rl: RL.comfort, rb: RB.comfort, better: true, suffix: "", showPct: true },
            { label: "Deadline ihlali", rl: RL.deadline, rb: RB.deadline, better: false, suffix: "", same: true },
            { label: "HVAC anahtar sayısı", rl: RL.hvac, rb: RB.hvac, better: true, suffix: "", showPct: true },
            { label: "Geçersiz aksiyon", rl: RL.invalid, rb: RB.invalid, better: false, suffix: "", same: true },
            { label: "Ortalama reward", rl: RL.reward, rb: RB.reward, better: true, suffix: "", showPct: false },
          ].map(row => (
            <div key={row.label} style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr 1fr", gap: 8, padding: "7px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)", alignItems: "center" }}>
              <span style={{ fontSize: 12, color: "rgba(240,244,248,0.55)" }}>{row.label}</span>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
                <span style={{ fontSize: 14, fontWeight: 500, color: row.same ? "rgba(240,244,248,0.4)" : "#5DCAA5" }}>{row.rl}</span>
                {row.showPct && (
                  <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 3, background: "rgba(29,158,117,0.15)", color: "#5DCAA5" }}>
                    -{pct(row.rl, row.rb)}%
                  </span>
                )}
              </div>
              <span style={{ fontSize: 14, fontWeight: 500, color: row.same ? "rgba(240,244,248,0.4)" : "rgba(240,244,248,0.6)", textAlign: "center" }}>{row.rb}</span>
            </div>
          ))}
        </div>

        {/* Alt satır */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>

          {/* Bar karşılaştırma */}
          <div style={{ background: "rgba(255,255,255,0.02)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: ".85rem 1rem" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Görsel karşılaştırma</div>
            {[
              { label: "Maliyet (TL)", rlW: 48, rbW: 50, vals: `${RL.cost} vs ${RB.cost}` },
              { label: "Konfor ihlali", rlW: 25, rbW: 50, vals: `${RL.comfort} vs ${RB.comfort}` },
              { label: "HVAC anahtarı", rlW: 32, rbW: 50, vals: `${RL.hvac} vs ${RB.hvac}` },
              { label: "Ort. reward", rlW: 44, rbW: 35, vals: `${RL.reward} vs ${RB.reward}` },
            ].map(b => (
              <div key={b.label} style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 11, color: "rgba(240,244,248,0.45)" }}>{b.label}</span>
                  <span style={{ fontSize: 10, color: "rgba(240,244,248,0.3)" }}>{b.vals}</span>
                </div>
                <div style={{ height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden", display: "flex", gap: 2 }}>
                  <div style={{ height: "100%", width: `${b.rlW}%`, background: "#1D9E75", borderRadius: 3 }} />
                  <div style={{ height: "100%", width: `${b.rbW}%`, background: "rgba(240,244,248,0.2)", borderRadius: 3 }} />
                </div>
              </div>
            ))}
          </div>

          {/* Bulgular */}
          <div style={{ background: "rgba(255,255,255,0.02)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 10, padding: ".85rem 1rem" }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Öne çıkan bulgular</div>
            {[
              { text: <span>RL ajanı <strong style={{ color: "#5DCAA5" }}>%{pct(RL.cost, RB.cost)} daha ucuz</strong> enerji kullandı.</span>, green: true },
              { text: <span>Konfor ihlali <strong style={{ color: "#5DCAA5" }}>%{pct(RL.comfort, RB.comfort)} azaldı</strong> — HVAC daha stabil çalıştı.</span>, green: true },
              { text: <span>Her iki ajan da <strong style={{ color: "#5DCAA5" }}>deadline ihlali yaşamadı</strong>.</span>, green: true },
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
          100 özdeş senaryo üzerinde paired evaluation · seed=2026
        </div>
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}