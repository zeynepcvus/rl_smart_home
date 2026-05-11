import { useState } from "react";

const SIDEBAR_STEPS = [
  { key: "welcome", label: "Hoşgeldiniz" },
  { key: "schedule", label: "Kullanıcı programı" },
  { key: "comfort", label: "Konfor tercihleri" },
  { key: "devices", label: "Cihaz seçimi" },
  { key: "summary", label: "Özet & onayla" },
];

export default function UserSchedule({ goTo, formData, updateForm }) {
  const [occupancy, setOccupancy] = useState(formData.occupancy || "home");

  const handleNext = () => {
    updateForm({ occupancy });
    goTo("comfort");
  };

  return (
    <div style={{
      minHeight: "100vh", background: "#0a1628", display: "flex",
      fontFamily: "'DM Sans', sans-serif"
    }}>
      {/* Grid */}
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
          <div style={{
            width: 7, height: 7, borderRadius: "50%", background: "#1D9E75",
            animation: "blink 2s ease infinite"
          }} />
          SmartHome RL
        </div>

        {SIDEBAR_STEPS.map((step, i) => {
          const isDone = i === 0;
          const isActive = step.key === "schedule";
          return (
            <div key={step.key} style={{
              display: "flex", alignItems: "center", gap: 10,
              padding: "8px 10px", borderRadius: 8,
              background: isActive ? "rgba(29,158,117,0.12)" : "transparent"
            }}>
              <div style={{
                width: 22, height: 22, borderRadius: "50%", flexShrink: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11,
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
      <div style={{
        flex: 1, padding: "2rem", position: "relative", zIndex: 1,
        display: "flex", flexDirection: "column"
      }}>
        {/* Progress bar */}
        <div style={{
          height: 2, background: "rgba(255,255,255,0.07)", borderRadius: 2,
          marginBottom: "2rem", overflow: "hidden"
        }}>
          <div style={{ height: "100%", width: "28%", background: "#1D9E75", borderRadius: 2 }} />
        </div>

        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem" }}>
          Günlük programın nedir?
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.4)", marginBottom: "1.75rem", fontWeight: 300 }}>
          Sistem, uyanma ve uyuma saatlerine göre cihaz kararlarını optimize eder.
        </p>

        {/* Uyku programı */}
        <div style={{
          background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
          borderRadius: 12, padding: "1.25rem", marginBottom: "1rem"
        }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase", marginBottom: "1rem" }}>
            Uyku programı
          </div>

          {[
            { label: "Uyanma saati", key: "awakeStart", options: [6, 7, 8, 9] },
            { label: "Uyuma saati", key: "sleepStart", options: [22, 23, 24] },
          ].map(field => (
            <div key={field.key} style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: ".85rem" }}>
              <span style={{ fontSize: 13, color: "rgba(240,244,248,0.6)", width: 110, flexShrink: 0 }}>{field.label}</span>
              <select
                value={formData[field.key]}
                onChange={e => updateForm({ [field.key]: parseInt(e.target.value) })}
                style={{
                  background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)",
                  borderRadius: 8, padding: "7px 12px", fontSize: 13, color: "#f0f4f8",
                  fontFamily: "'DM Sans', sans-serif", cursor: "pointer", width: 110
                }}
              >
                {field.options.map(o => (
                  <option key={o} value={o}>{String(o).padStart(2, "0")}:00</option>
                ))}
              </select>
            </div>
          ))}
        </div>

        {/* Evde bulunma */}
        <div style={{
          background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
          borderRadius: 12, padding: "1.25rem", marginBottom: "1rem"
        }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase", marginBottom: "1rem" }}>
            Evde bulunma durumu
          </div>
          <div style={{ fontSize: 14, color: "#f0f4f8", marginBottom: "1rem" }}>
            Bugün gün içinde evde olacak mısınız?
          </div>

          <div style={{ display: "flex", gap: 10, marginBottom: "1rem" }}>
            {[
              { val: "home", label: "Evet, gün boyu evdeyim" },
              { val: "partial", label: "Kısmen — belirli saatlerde yokum" },
              { val: "away", label: "Hayır, gün boyu dışarıdayım" },
            ].map(opt => (
              <button
                key={opt.val}
                onClick={() => setOccupancy(opt.val)}
                style={{
                  flex: 1, padding: 10, borderRadius: 8, cursor: "pointer",
                  fontFamily: "'DM Sans', sans-serif", fontSize: 13, textAlign: "center",
                  background: occupancy === opt.val ? "rgba(29,158,117,0.12)" : "transparent",
                  border: occupancy === opt.val ? "0.5px solid #1D9E75" : "0.5px solid rgba(255,255,255,0.1)",
                  color: occupancy === opt.val ? "#5DCAA5" : "rgba(240,244,248,0.5)",
                  fontWeight: occupancy === opt.val ? 500 : 400,
                  transition: "all .2s"
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {occupancy === "partial" && (
            <div style={{
              display: "flex", alignItems: "center", gap: "1rem", padding: ".75rem 1rem",
              background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
              borderRadius: 8
            }}>
              <span style={{ fontSize: 13, color: "rgba(240,244,248,0.6)", flex: 1 }}>Evde olmadığım saatler</span>
              <select
                value={formData.awayFrom}
                onChange={e => updateForm({ awayFrom: parseInt(e.target.value) })}
                style={{
                  background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)",
                  borderRadius: 6, padding: "5px 10px", fontSize: 12, color: "#f0f4f8",
                  fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
                }}
              >
                {[8,9,10,11,12].map(h => <option key={h} value={h}>{h}:00</option>)}
              </select>
              <span style={{ fontSize: 12, color: "rgba(240,244,248,0.35)" }}>—</span>
              <select
                value={formData.awayTo}
                onChange={e => updateForm({ awayTo: parseInt(e.target.value) })}
                style={{
                  background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.12)",
                  borderRadius: 6, padding: "5px 10px", fontSize: 12, color: "#f0f4f8",
                  fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
                }}
              >
                {[16,17,18,19,20].map(h => <option key={h} value={h}>{h}:00</option>)}
              </select>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1.25rem" }}>
          <button
            onClick={() => goTo("welcome")}
            style={{
              background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)",
              borderRadius: 8, padding: "10px 20px", fontSize: 13,
              color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
            }}
          >← Geri</button>
          <button
            onClick={handleNext}
            style={{
              background: "#1D9E75", border: "none", borderRadius: 8,
              padding: "10px 28px", fontSize: 13, fontWeight: 500,
              color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
            }}
          >Devam Et →</button>
        </div>
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} } select option { background: #0a1628; }`}</style>
    </div>
  );
}