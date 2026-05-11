const SIDEBAR_STEPS = [
  { key: "welcome", label: "Hoşgeldiniz" },
  { key: "schedule", label: "Kullanıcı programı" },
  { key: "comfort", label: "Konfor tercihleri" },
  { key: "devices", label: "Cihaz seçimi" },
  { key: "summary", label: "Özet & onayla" },
];

const COLORS = [
  { bg: "rgba(29,158,117,0.15)", color: "#5DCAA5" },
  { bg: "rgba(55,138,221,0.15)", color: "#85B7EB" },
  { bg: "rgba(250,199,117,0.15)", color: "#FAC775" },
  { bg: "rgba(168,85,247,0.15)", color: "#c084fc" },
  { bg: "rgba(239,159,39,0.15)", color: "#EF9F27" },
];

export default function Summary({ goTo, formData }) {
  const occupancyLabel = {
    home: "Gün boyu evde",
    partial: `${formData.awayFrom}:00 – ${formData.awayTo}:00 arası dışarıda`,
    away: "Gün boyu dışarıda",
  }[formData.occupancy] || "—";

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
          const isDone = i < 4;
          const isActive = step.key === "summary";
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
          <div style={{ height: "100%", width: "100%", background: "#1D9E75", borderRadius: 2 }} />
        </div>

        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem" }}>
          Her şey doğru mu?
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.4)", marginBottom: "1.5rem", fontWeight: 300 }}>
          Girdiğin bilgileri kontrol et, sonra simülasyonu başlat.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: "1rem" }}>
          {/* Kullanıcı programı */}
          <div style={{
            background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
            borderRadius: 12, padding: "1rem 1.1rem"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".75rem" }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase" }}>Kullanıcı programı</span>
              <button onClick={() => goTo("schedule")} style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>Düzenle</button>
            </div>
            {[
              { label: "Uyanma saati", val: `${String(formData.awakeStart).padStart(2,"0")}:00` },
              { label: "Uyuma saati", val: `${String(formData.sleepStart).padStart(2,"0")}:00` },
              { label: "Evde bulunma", val: occupancyLabel },
            ].map(row => (
              <div key={row.label} style={{
                display: "flex", justifyContent: "space-between", padding: "4px 0",
                borderBottom: "0.5px solid rgba(255,255,255,0.05)"
              }}>
                <span style={{ fontSize: 12, color: "rgba(240,244,248,0.45)" }}>{row.label}</span>
                <span style={{ fontSize: 12, color: "#f0f4f8", fontWeight: 500 }}>{row.val}</span>
              </div>
            ))}
          </div>

          {/* Konfor tercihleri */}
          <div style={{
            background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
            borderRadius: 12, padding: "1rem 1.1rem"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".75rem" }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase" }}>Konfor tercihleri</span>
              <button onClick={() => goTo("comfort")} style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>Düzenle</button>
            </div>
            {[
              { label: "Min sıcaklık", val: `${formData.tempMin} °C` },
              { label: "Max sıcaklık", val: `${formData.tempMax} °C` },
              { label: "Aydınlatma kontrolü", val: formData.lightingEnabled ? "Aktif" : "Pasif", green: formData.lightingEnabled },
            ].map(row => (
              <div key={row.label} style={{
                display: "flex", justifyContent: "space-between", padding: "4px 0",
                borderBottom: "0.5px solid rgba(255,255,255,0.05)"
              }}>
                <span style={{ fontSize: 12, color: "rgba(240,244,248,0.45)" }}>{row.label}</span>
                <span style={{ fontSize: 12, color: row.green ? "#5DCAA5" : "#f0f4f8", fontWeight: 500 }}>{row.val}</span>
              </div>
            ))}
          </div>

          {/* Cihazlar */}
          <div style={{
            gridColumn: "1 / -1",
            background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)",
            borderRadius: 12, padding: "1rem 1.1rem"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".75rem" }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase" }}>
                Cihazlar ({formData.devices.length} / 5 slot)
              </span>
              <button onClick={() => goTo("devices")} style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>Düzenle</button>
            </div>
            {formData.devices.length === 0 ? (
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.3)", textAlign: "center", padding: ".5rem" }}>Henüz cihaz eklenmedi</div>
            ) : formData.devices.map((d, i) => {
              const c = COLORS[i % COLORS.length];
              const initials = d.name.split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();
              return (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 8, padding: "5px 0",
                  borderBottom: "0.5px solid rgba(255,255,255,0.05)"
                }}>
                  <div style={{
                    width: 24, height: 24, borderRadius: 6, background: c.bg, color: c.color,
                    display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 500, flexShrink: 0
                  }}>{initials}</div>
                  <span style={{ fontSize: 12, color: "#f0f4f8", flex: 1 }}>{d.name}</span>
                  <span style={{
                    fontSize: 9, padding: "1px 6px", borderRadius: 4, fontWeight: 500,
                    background: d.type === "shiftable" ? "rgba(55,138,221,0.15)" : "rgba(29,158,117,0.15)",
                    color: d.type === "shiftable" ? "#85B7EB" : "#5DCAA5"
                  }}>{d.type === "shiftable" ? "Ertelenebilir" : "Sürekli"}</span>
                  <span style={{ fontSize: 11, color: "rgba(240,244,248,0.4)" }}>{d.power} kW</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Onay kutusu */}
        <div style={{
          background: "rgba(29,158,117,0.07)", border: "0.5px solid rgba(29,158,117,0.25)",
          borderRadius: 12, padding: "1rem 1.1rem", marginBottom: "1rem",
          display: "flex", alignItems: "center", gap: 12
        }}>
          <div style={{
            width: 36, height: 36, borderRadius: "50%", background: "rgba(29,158,117,0.15)",
            display: "flex", alignItems: "center", justifyContent: "center", color: "#5DCAA5", fontSize: 16, flexShrink: 0
          }}>✓</div>
          <div>
            <div style={{ fontSize: 13, color: "#5DCAA5", fontWeight: 500, marginBottom: 2 }}>Simülasyon başlatılmaya hazır</div>
            <div style={{ fontSize: 12, color: "rgba(240,244,248,0.55)" }}>
              {formData.devices.length} cihaz, 24 saatlik optimizasyon · RL ajanı ve kural tabanlı ajan aynı anda çalışacak
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1rem" }}>
          <button onClick={() => goTo("devices")} style={{
            background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)",
            borderRadius: 8, padding: "10px 20px", fontSize: 13,
            color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer"
          }}>← Geri</button>
          <button
            onClick={() => goTo("dashboard")}
            style={{
              background: "#1D9E75", border: "none", borderRadius: 8,
              padding: "11px 32px", fontSize: 14, fontWeight: 500,
              color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer",
              display: "flex", alignItems: "center", gap: 10
            }}
          >
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "rgba(255,255,255,0.5)", animation: "blink 1.5s ease infinite" }} />
            Simülasyonu Başlat
          </button>
        </div>
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}