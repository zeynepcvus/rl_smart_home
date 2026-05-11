import { useState } from "react";

const SIDEBAR_STEPS = [
  { key: "welcome", label: "Hoşgeldiniz" },
  { key: "schedule", label: "Kullanıcı programı" },
  { key: "comfort", label: "Konfor tercihleri" },
  { key: "devices", label: "Cihaz seçimi" },
  { key: "summary", label: "Özet & onayla" },
];

export default function ComfortPrefs({ goTo, formData, updateForm }) {
  const [lighting, setLighting] = useState(formData.lightingEnabled);

  const handleNext = () => {
    updateForm({ lightingEnabled: lighting });
    goTo("devices");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      {/* Sidebar */}
      <div style={{
        width: 200, flexShrink: 0, borderRight: "0.5px solid rgba(255,255,255,0.07)",
        padding: "1.5rem 1.25rem", display: "flex", flexDirection: "column",
        gap: ".25rem", position: "relative", zIndex: 1
      }}>
        <div style={{
          fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5",
          marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: 6
        }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#1D9E75", animation: "blink 2s ease infinite" }} />
          SmartHome RL
        </div>
        {SIDEBAR_STEPS.map((step, i) => {
          const isDone = i < 2;
          const isActive = step.key === "comfort";
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
      </div>

      {/* Main */}
      <div style={{ flex: 1, padding: "2rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column" }}>
        <div style={{ height: 2, background: "rgba(255,255,255,0.07)", borderRadius: 2, marginBottom: "2rem", overflow: "hidden" }}>
          <div style={{ height: "100%", width: "56%", background: "#1D9E75", borderRadius: 2 }} />
        </div>

        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem" }}>
          Konfor tercihlerini belirle
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.4)", marginBottom: "1.75rem", fontWeight: 300 }}>
          HVAC sistemi bu sınırlar içinde iç sıcaklığı koruyacak şekilde çalışır.
        </p>

        {/* Sıcaklık bandı */}
        <div style={{
          background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
          borderRadius: 12, padding: "1.25rem", marginBottom: "1rem"
        }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase", marginBottom: "1.25rem" }}>
            Sıcaklık bandı
          </div>

          {[
            { label: "Minimum sıcaklık", key: "tempMin", min: 16, max: 22 },
            { label: "Maksimum sıcaklık", key: "tempMax", min: 22, max: 30 },
          ].map(field => (
            <div key={field.key} style={{ marginBottom: "1.25rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".6rem" }}>
                <span style={{ fontSize: 13, color: "rgba(240,244,248,0.7)" }}>{field.label}</span>
                <span style={{ fontSize: 15, fontWeight: 500, color: "#5DCAA5", fontFamily: "'DM Serif Display', serif" }}>
                  {formData[field.key]} °C
                </span>
              </div>
              <input
                type="range"
                min={field.min}
                max={field.max}
                value={formData[field.key]}
                onChange={e => {
                  const val = parseInt(e.target.value);
                  if (field.key === "tempMin" && val >= formData.tempMax) return;
                  if (field.key === "tempMax" && val <= formData.tempMin) return;
                  updateForm({ [field.key]: val });
                }}
                style={{ width: "100%", accentColor: "#1D9E75", cursor: "pointer" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "rgba(240,244,248,0.25)", marginTop: 4 }}>
                <span>{field.min} °C</span>
                <span>{field.max} °C</span>
              </div>
            </div>
          ))}

          {/* Bant önizleme */}
          <div style={{
            display: "flex", alignItems: "center", gap: "1rem", padding: ".75rem 1rem",
            background: "rgba(29,158,117,0.07)", border: "0.5px solid rgba(29,158,117,0.2)", borderRadius: 8
          }}>
            <span style={{ fontSize: 12, color: "rgba(240,244,248,0.5)", whiteSpace: "nowrap" }}>16°C</span>
            <div style={{ flex: 1, height: 8, background: "rgba(255,255,255,0.07)", borderRadius: 4, position: "relative", overflow: "hidden" }}>
              <div style={{
                position: "absolute", height: "100%", background: "#1D9E75", borderRadius: 4,
                left: `${((formData.tempMin - 16) / 14) * 100}%`,
                width: `${((formData.tempMax - formData.tempMin) / 14) * 100}%`,
                transition: "left .3s, width .3s"
              }} />
            </div>
            <span style={{ fontSize: 12, color: "rgba(240,244,248,0.5)", whiteSpace: "nowrap" }}>30°C</span>
            <span style={{ fontSize: 12, color: "#5DCAA5", fontWeight: 500, whiteSpace: "nowrap" }}>
              Hedef: {formData.tempMin}–{formData.tempMax} °C
            </span>
          </div>
        </div>

        {/* Aydınlatma toggle */}
        <div style={{
          background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
          borderRadius: 12, padding: "1.25rem", marginBottom: "1rem"
        }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase", marginBottom: "1rem" }}>
            Aydınlatma
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, color: "rgba(240,244,248,0.85)", marginBottom: 2 }}>Aydınlatma konfor kontrolü</div>
              <div style={{ fontSize: 11, color: "rgba(240,244,248,0.35)", lineHeight: 1.5 }}>
                Gece ve kullanıcı evdeyken aydınlatma ihtiyacı hesaba katılır.
              </div>
            </div>
            <div
              onClick={() => setLighting(!lighting)}
              style={{
                width: 40, height: 22, borderRadius: 11, cursor: "pointer",
                background: lighting ? "#1D9E75" : "rgba(255,255,255,0.1)",
                position: "relative", transition: "background .2s", flexShrink: 0
              }}
            >
              <div style={{
                position: "absolute", width: 16, height: 16, borderRadius: "50%",
                background: "#fff", top: 3,
                left: lighting ? 21 : 3, transition: "left .2s"
              }} />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1.25rem" }}>
          <button onClick={() => goTo("schedule")} style={{
            background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)",
            borderRadius: 8, padding: "10px 20px", fontSize: 13,
            color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
          }}>← Geri</button>
          <button onClick={handleNext} style={{
            background: "#1D9E75", border: "none", borderRadius: 8,
            padding: "10px 28px", fontSize: 13, fontWeight: 500,
            color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
          }}>Devam Et →</button>
        </div>
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} } select option { background: #0a1628; }`}</style>
    </div>
  );
}