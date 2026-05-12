const MODES = [
  { id: "cost", icon: "💰", label: "Maliyet Odaklı", desc: "Elektrik faturanı minimize et. Konfor ikinci planda." },
  { id: "balanced", icon: "⚖️", label: "Dengeli", desc: "Maliyet ve konfor arasında optimal denge." },
  { id: "comfort", icon: "🌡️", label: "Konfor Odaklı", desc: "Maksimum konfor. Maliyet ikinci planda." },
];

export default function OptimizationMode({ goTo, formData, updateForm }) {
  const selected = formData.mode || "balanced";

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{ position: "fixed", inset: 0, backgroundImage: "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)", backgroundSize: "40px 40px", pointerEvents: "none" }} />

      <div style={{ flex: 1, padding: "2rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column", maxWidth: 600, margin: "0 auto" }}>
        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem", marginTop: "2rem" }}>
          Optimizasyon modunu seç
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.4)", marginBottom: "2rem", fontWeight: 300 }}>
          Sistem hangi hedefe öncelik versin?
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: "auto" }}>
          {MODES.map(m => (
            <div
              key={m.id}
              onClick={() => updateForm({ mode: m.id })}
              style={{
                background: selected === m.id ? "rgba(29,158,117,0.1)" : "rgba(255,255,255,0.03)",
                border: selected === m.id ? "1.5px solid #1D9E75" : "0.5px solid rgba(255,255,255,0.08)",
                borderRadius: 12, padding: "1.1rem 1.25rem", cursor: "pointer",
                display: "flex", alignItems: "center", gap: 16, transition: "all 0.15s"
              }}
            >
              <span style={{ fontSize: 28 }}>{m.icon}</span>
              <div>
                <div style={{ fontSize: 14, fontWeight: 500, color: selected === m.id ? "#5DCAA5" : "#f0f4f8", marginBottom: 3 }}>{m.label}</div>
                <div style={{ fontSize: 12, color: "rgba(240,244,248,0.4)" }}>{m.desc}</div>
              </div>
              {selected === m.id && (
                <div style={{ marginLeft: "auto", width: 18, height: 18, borderRadius: "50%", background: "#1D9E75", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, color: "#fff" }}>✓</div>
              )}
            </div>
          ))}
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", paddingTop: "2rem" }}>
          <button onClick={() => goTo("schedule")} style={{ background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 20px", fontSize: 13, color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>← Geri</button>
          <button onClick={() => goTo("comfort")} style={{ background: "#1D9E75", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>Devam Et →</button>
        </div>
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
