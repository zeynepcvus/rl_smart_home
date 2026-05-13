const STEPS = [
  { key: "welcome", label: "Hoşgeldiniz" },
  { key: "schedule", label: "Kullanıcı programı" },
  { key: "mode", label: "Optimizasyon modu" },
  { key: "devices", label: "Cihaz seçimi" },
  { key: "comfort", label: "Konfor tercihleri" },
  { key: "summary", label: "Özet & onayla" },
];

export default function Sidebar({ currentStep }) {
  const currentIndex = STEPS.findIndex(s => s.key === currentStep);

  return (
    <div style={{
      width: 210, flexShrink: 0, borderRight: "0.5px solid rgba(255,255,255,0.07)",
      padding: "1.5rem 1.25rem", display: "flex", flexDirection: "column",
      position: "relative", zIndex: 1
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: "2rem" }}>
        <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#1D9E75", boxShadow: "0 0 6px rgba(29,158,117,0.8)", animation: "blink 2s ease infinite", flexShrink: 0 }} />
        <span style={{ fontFamily: "'DM Serif Display', serif", fontSize: 15, color: "#f0f4f8" }}>SmartHome</span>
        <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 11, fontWeight: 700, color: "#1D9E75", background: "rgba(29,158,117,0.15)", border: "0.5px solid rgba(29,158,117,0.4)", borderRadius: 5, padding: "1px 6px", letterSpacing: ".04em" }}>RL</span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        {STEPS.map((step, i) => {
          const isDone = i < currentIndex;
          const isActive = i === currentIndex;
          const isLast = i === STEPS.length - 1;
          return (
            <div key={step.key} style={{ display: "flex", gap: 10, position: "relative" }}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0 }}>
                <div style={{
                  width: 24, height: 24, borderRadius: "50%", flexShrink: 0,
                  display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 600,
                  background: isDone ? "#1D9E75" : isActive ? "#1D9E75" : "rgba(255,255,255,0.05)",
                  border: isDone ? "none" : isActive ? "none" : "0.5px solid rgba(255,255,255,0.15)",
                  color: isDone ? "#fff" : isActive ? "#fff" : "rgba(240,244,248,0.3)",
                  boxShadow: isActive ? "0 0 8px rgba(29,158,117,0.5)" : "none",
                  zIndex: 1
                }}>
                  {isDone ? "✓" : i + 1}
                </div>
                {!isLast && (
                  <div style={{ width: 1.5, flex: 1, minHeight: 18, background: isDone ? "rgba(29,158,117,0.5)" : "rgba(255,255,255,0.08)", marginTop: 2, marginBottom: 2 }} />
                )}
              </div>
              <div style={{
                flex: 1, padding: "4px 8px 18px", borderRadius: 8,
                background: isActive ? "rgba(29,158,117,0.1)" : "transparent",
                marginBottom: isLast ? 0 : 0
              }}>
                <span style={{
                  fontSize: 12,
                  color: isDone ? "#5DCAA5" : isActive ? "#f0f4f8" : "rgba(240,244,248,0.3)",
                  fontWeight: isActive ? 600 : 400,
                  lineHeight: "24px"
                }}>{step.label}</span>
              </div>
            </div>
          );
        })}
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}