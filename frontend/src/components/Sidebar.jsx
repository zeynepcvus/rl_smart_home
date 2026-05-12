const STEPS = [
  { id: "welcome", label: "Hoşgeldiniz", step: null },
  { id: "schedule", label: "Kullanıcı programı", step: 2 },
  { id: "mode", label: "Optimizasyon modu", step: 3 },
  { id: "comfort", label: "Konfor tercihleri", step: 3 },
  { id: "devices", label: "Cihaz seçimi", step: 4 },
  { id: "summary", label: "Özet & onayla", step: 5 },
];

export default function Sidebar({ currentStep }) {
  const steps = [
    { label: "Hoşgeldiniz", done: true },
    { label: "Kullanıcı programı", done: ["comfort","mode","devices","summary"].includes(currentStep) },
    { label: "Konfor tercihleri", done: ["devices","summary"].includes(currentStep) },
    { label: "Cihaz seçimi", done: ["summary"].includes(currentStep) },
    { label: "Özet & onayla", done: false },
  ];

  return (
    <div style={{ width: 200, minHeight: "100vh", borderRight: "0.5px solid rgba(255,255,255,0.07)", padding: "1.5rem 1rem", display: "flex", flexDirection: "column", gap: 8, position: "relative", zIndex: 1 }}>
      <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5", marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#1D9E75" }} />
        SmartHome RL
      </div>
      {steps.map((s, i) => {
        const active = (i === 0 && currentStep === "welcome") ||
          (i === 1 && currentStep === "schedule") ||
          (i === 2 && (currentStep === "comfort" || currentStep === "mode")) ||
          (i === 3 && currentStep === "devices") ||
          (i === 4 && currentStep === "summary");
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 8px", borderRadius: 8, background: active ? "rgba(29,158,117,0.1)" : "transparent" }}>
            <div style={{ width: 20, height: 20, borderRadius: "50%", background: s.done ? "#1D9E75" : active ? "rgba(29,158,117,0.3)" : "rgba(255,255,255,0.07)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "#fff", flexShrink: 0 }}>
              {s.done ? "✓" : i + 1}
            </div>
            <span style={{ fontSize: 12, color: active ? "#5DCAA5" : s.done ? "rgba(240,244,248,0.6)" : "rgba(240,244,248,0.3)" }}>{s.label}</span>
          </div>
        );
      })}
    </div>
  );
}
