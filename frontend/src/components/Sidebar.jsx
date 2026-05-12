const STEPS = [
  { key: "welcome", label: "Hoşgeldiniz" },
  { key: "schedule", label: "Kullanıcı programı" },
  { key: "mode", label: "Optimizasyon modu" },
  { key: "comfort", label: "Konfor tercihleri" },
  { key: "devices", label: "Cihaz seçimi" },
  { key: "summary", label: "Özet & onayla" },
];

export default function Sidebar({ currentStep }) {
  const currentIndex = STEPS.findIndex(s => s.key === currentStep);

  return (
    <div style={{
      width: 200, flexShrink: 0, borderRight: "0.5px solid rgba(255,255,255,0.07)",
      padding: "1.5rem 1.25rem", display: "flex", flexDirection: "column",
      gap: ".25rem", position: "relative", zIndex: 1
    }}>
      <div style={{
        fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5",
        marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: 6
      }}>
        <div style={{
          width: 7, height: 7, borderRadius: "50%", background: "#1D9E75",
          animation: "blink 2s ease infinite"
        }} />
        SmartHome RL
      </div>

      {STEPS.map((step, i) => {
        const isDone = i < currentIndex;
        const isActive = i === currentIndex;
        return (
          <div key={step.key} style={{
            display: "flex", alignItems: "center", gap: 10, padding: "8px 10px",
            borderRadius: 8, background: isActive ? "rgba(29,158,117,0.12)" : "transparent"
          }}>
            <div style={{
              width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11,
              background: isDone ? "rgba(29,158,117,0.2)" : isActive ? "#1D9E75" : "transparent",
              border: isDone ? "0.5px solid #1D9E75" : isActive ? "none" : "0.5px solid rgba(255,255,255,0.15)",
              color: isDone ? "#5DCAA5" : isActive ? "#fff" : "rgba(240,244,248,0.35)"
            }}>
              {isDone ? "✓" : i + 1}
            </div>
            <span style={{
              fontSize: 12,
              color: isDone ? "#5DCAA5" : isActive ? "#f0f4f8" : "rgba(240,244,248,0.3)",
              fontWeight: isActive ? 500 : 400
            }}>{step.label}</span>
          </div>
        );
      })}
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}