import Sidebar from "../components/Sidebar";

const modes = [
  {
    key: "cost",
    icon: "💰",
    title: "Maliyet Odaklı",
    desc: "Elektrik faturanı agresif şekilde düşür. Konfor ikinci planda kalabilir.",
    detail: "HVAC daha az çalışır, cihazlar gece saatlerine kaydırılır.",
    stats: { cost: "10.83 TL", comfort: "9.40 ihlal", improvement: "%65.5 tasarruf" },
    color: "#85B7EB",
    bg: "rgba(55,138,221,0.08)",
    border: "rgba(55,138,221,0.3)",
  },
  {
    key: "balanced",
    icon: "⚖️",
    title: "Dengeli",
    desc: "Maliyet ve konfor arasında en iyi dengeyi kur. Önerilen mod.",
    detail: "Hem faturanı düşürür hem evi konforlu tutar.",
    stats: { cost: "26.39 TL", comfort: "2.12 ihlal", improvement: "%15.2 tasarruf" },
    color: "#5DCAA5",
    bg: "rgba(29,158,117,0.08)",
    border: "#1D9E75",
    recommended: true,
  },
  {
    key: "comfort",
    icon: "🌡️",
    title: "Konfor Odaklı",
    desc: "Evin her zaman konforlu olsun. Maliyet ikinci planda kalabilir.",
    detail: "HVAC sürekli aktif, sıcaklık bandı sıkı korunur.",
    stats: { cost: "35.41 TL", comfort: "1.77 ihlal", improvement: "Konfor öncelikli" },
    color: "#FAC775",
    bg: "rgba(250,199,117,0.08)",
    border: "rgba(250,199,117,0.3)",
  },
];

export default function OptimizationMode({ goTo, formData, updateForm }) {
  const selected = formData.mode || "balanced";

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      <Sidebar currentStep="mode" />

      <div style={{ flex: 1, padding: "2rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column" }}>
        <div style={{ height: 2, background: "rgba(255,255,255,0.07)", borderRadius: 2, marginBottom: "2rem", overflow: "hidden" }}>
          <div style={{ height: "100%", width: "42%", background: "#1D9E75", borderRadius: 2 }} />
        </div>

        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem" }}>
          Optimizasyon modunu seç
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.4)", marginBottom: "1.75rem", fontWeight: 300 }}>
          Sistem bu tercihe göre kararlarını optimize eder.
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: "1rem" }}>
          {modes.map(mode => (
            <div
              key={mode.key}
              onClick={() => updateForm({ mode: mode.key })}
              style={{
                background: selected === mode.key ? mode.bg : "rgba(255,255,255,0.02)",
                border: selected === mode.key ? `1.5px solid ${mode.border}` : "0.5px solid rgba(255,255,255,0.08)",
                borderRadius: 12, padding: "1.1rem 1.25rem", cursor: "pointer",
                transition: "all .2s", position: "relative"
              }}
            >
              {mode.recommended && (
                <div style={{
                  position: "absolute", top: 12, right: 12, fontSize: 10,
                  padding: "2px 8px", borderRadius: 20,
                  background: "rgba(29,158,117,0.2)", color: "#5DCAA5", fontWeight: 500
                }}>Önerilen</div>
              )}
              <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
                <div style={{ fontSize: 28, flexShrink: 0, marginTop: 2 }}>{mode.icon}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 15, fontWeight: 500, color: selected === mode.key ? mode.color : "#f0f4f8", marginBottom: 4 }}>
                    {mode.title}
                  </div>
                  <div style={{ fontSize: 13, color: "rgba(240,244,248,0.55)", marginBottom: 8, lineHeight: 1.5 }}>
                    {mode.desc}
                  </div>
                  <div style={{ fontSize: 11, color: "rgba(240,244,248,0.35)", marginBottom: 10 }}>
                    {mode.detail}
                  </div>
                  <div style={{ display: "flex", gap: 16 }}>
                    {[
                      { label: "Ort. maliyet", val: mode.stats.cost },
                      { label: "Konfor ihlali", val: mode.stats.comfort },
                      { label: "Performans", val: mode.stats.improvement },
                    ].map(s => (
                      <div key={s.label}>
                        <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginBottom: 2 }}>{s.label}</div>
                        <div style={{ fontSize: 13, fontWeight: 500, color: mode.color }}>{s.val}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div style={{
                  width: 20, height: 20, borderRadius: "50%", flexShrink: 0, marginTop: 2,
                  border: selected === mode.key ? `2px solid ${mode.color}` : "1.5px solid rgba(255,255,255,0.2)",
                  background: selected === mode.key ? mode.color : "transparent",
                  display: "flex", alignItems: "center", justifyContent: "center"
                }}>
                  {selected === mode.key && <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#fff" }} />}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1rem" }}>
          <button onClick={() => goTo("schedule")} style={{ background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 20px", fontSize: 13, color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>← Geri</button>
          <button onClick={() => goTo("devices")} style={{ background: "#1D9E75", border: "none", borderRadius: 8, padding: "10px 28px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>Devam Et →</button>
        </div>
      </div>
    </div>
  );
}