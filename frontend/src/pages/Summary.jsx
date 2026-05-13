import { useState } from "react";
import Sidebar from "../components/Sidebar";

const COLORS = [
  { bg: "rgba(29,158,117,0.15)", color: "#5DCAA5" },
  { bg: "rgba(55,138,221,0.15)", color: "#85B7EB" },
  { bg: "rgba(250,199,117,0.15)", color: "#FAC775" },
  { bg: "rgba(168,85,247,0.15)", color: "#c084fc" },
  { bg: "rgba(239,159,39,0.15)", color: "#EF9F27" },
];

const MODE_LABELS = {
  cost: { label: "Maliyet Odaklı", icon: "💰", color: "#85B7EB" },
  balanced: { label: "Dengeli", icon: "⚖️", color: "#5DCAA5" },
  comfort: { label: "Konfor Odaklı", icon: "🌡️", color: "#FAC775" },
};

export default function Summary({ goTo, formData, setApiResult, saveProfile }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const occupancyLabel = {
    home: "Gün boyu evde",
    partial: `${formData.awayFrom}:00 – ${formData.awayTo}:00 arası dışarıda`,
    away: "Gün boyu dışarıda",
  }[formData.occupancy] || "—";

  const mode = MODE_LABELS[formData.mode] || MODE_LABELS.balanced;

  const handleStart = async () => {
    setLoading(true);
    setError(null);
    try {
      const devices = formData.devices
        .filter(d => d.activeToday !== false && d.apiName !== "HVAC" && d.apiName !== "Lighting")
        .map(d => ({
          name: d.apiName ?? d.name,
          preset: d.apiName != null,
          power_kw: d.power ?? null,
          duration: d.duration !== "" ? d.duration : null,
          deadline: d.deadline !== "" ? d.deadline : null,
        }));

      const res = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: formData.mode,
          user_home: formData.occupancy !== "away",
          temp_min: formData.tempMin,
          temp_max: formData.tempMax,
          awake_start: formData.awakeStart,
          sleep_start: formData.sleepStart,
          devices,
        }),
      });

      if (!res.ok) throw new Error("API hatası: " + res.status);
      const data = await res.json();
      setApiResult(data);
      saveProfile();
      goTo("dashboard");
    } catch (err) {
      setError("Sunucuya bağlanılamadı. API çalışıyor mu?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      <Sidebar currentStep="summary" />

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
          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderLeft: "3px solid #85B7EB", borderRadius: 12, padding: "1rem 1.1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".75rem" }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "#85B7EB", letterSpacing: ".07em", textTransform: "uppercase" }}>Kullanıcı programı</span>
              <button onClick={() => goTo("schedule")} style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>Düzenle</button>
            </div>
            {[
              { label: "Uyanma saati", val: `${String(formData.awakeStart).padStart(2,"0")}:00` },
              { label: "Uyuma saati", val: `${String(formData.sleepStart).padStart(2,"0")}:00` },
              { label: "Evde bulunma", val: occupancyLabel },
            ].map(row => (
              <div key={row.label} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
                <span style={{ fontSize: 12, color: "rgba(240,244,248,0.45)" }}>{row.label}</span>
                <span style={{ fontSize: 12, color: "#f0f4f8", fontWeight: 500 }}>{row.val}</span>
              </div>
            ))}
          </div>

          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderLeft: "3px solid #1D9E75", borderRadius: 12, padding: "1rem 1.1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".75rem" }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "#5DCAA5", letterSpacing: ".07em", textTransform: "uppercase" }}>Konfor tercihleri</span>
              <button onClick={() => goTo("comfort")} style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>Düzenle</button>
            </div>
            {[
              { label: "Min sıcaklık", val: `${formData.tempMin} °C` },
              { label: "Max sıcaklık", val: `${formData.tempMax} °C` },
            ].map(row => (
              <div key={row.label} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
                <span style={{ fontSize: 12, color: "rgba(240,244,248,0.45)" }}>{row.label}</span>
                <span style={{ fontSize: 12, color: row.green ? "#5DCAA5" : "#f0f4f8", fontWeight: 500 }}>{row.val}</span>
              </div>
            ))}
          </div>

          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderLeft: `3px solid ${mode.color}`, borderRadius: 12, padding: "1rem 1.1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".75rem" }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: mode.color, letterSpacing: ".07em", textTransform: "uppercase" }}>Optimizasyon modu</span>
              <button onClick={() => goTo("mode")} style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>Düzenle</button>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0" }}>
              <span style={{ fontSize: 20 }}>{mode.icon}</span>
              <span style={{ fontSize: 14, fontWeight: 500, color: mode.color }}>{mode.label}</span>
            </div>
          </div>

          <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderLeft: "3px solid #c084fc", borderRadius: 12, padding: "1rem 1.1rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".75rem" }}>
              <span style={{ fontSize: 11, fontWeight: 500, color: "#c084fc", letterSpacing: ".07em", textTransform: "uppercase" }}>
                Cihazlar ({formData.devices.filter(d => d.activeToday !== false).length} aktif · {formData.devices.length} kayıtlı)
              </span>
              <button onClick={() => goTo("devices")} style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>Düzenle</button>
            </div>
            {formData.devices.length === 0 ? (
              <div style={{ fontSize: 12, color: "rgba(240,244,248,0.3)", textAlign: "center", padding: ".5rem" }}>Henüz cihaz eklenmedi</div>
            ) : formData.devices.map((d, i) => {
              const c = COLORS[i % COLORS.length];
              const initials = d.name.split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();
              const isActive = d.activeToday !== false;
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)", opacity: isActive ? 1 : 0.4 }}>
                  <div style={{ width: 24, height: 24, borderRadius: 6, background: c.bg, color: c.color, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 500, flexShrink: 0 }}>{initials}</div>
                  <span style={{ fontSize: 12, color: "#f0f4f8", flex: 1 }}>{d.name}</span>
                  {!isActive
                    ? <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, background: "rgba(255,255,255,0.07)", color: "rgba(240,244,248,0.4)" }}>Bugün pasif</span>
                    : <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, fontWeight: 500, background: d.type === "shiftable" ? "rgba(55,138,221,0.15)" : "rgba(29,158,117,0.15)", color: d.type === "shiftable" ? "#85B7EB" : "#5DCAA5" }}>
                        {d.type === "shiftable" ? "Ertelenebilir" : "Sürekli"}
                      </span>
                  }
                  <span style={{ fontSize: 11, color: "rgba(240,244,248,0.4)" }}>{d.power} kW</span>
                </div>
              );
            })}
          </div>
        </div>

        {error && (
          <div style={{ background: "rgba(226,75,74,0.1)", border: "0.5px solid rgba(226,75,74,0.3)", borderRadius: 10, padding: "10px 16px", marginBottom: "1rem", fontSize: 13, color: "#e24b4a" }}>
            {error}
          </div>
        )}

        <div style={{ background: "rgba(29,158,117,0.07)", border: "0.5px solid rgba(29,158,117,0.25)", borderRadius: 12, padding: "1rem 1.1rem", marginBottom: "1rem", display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 36, height: 36, borderRadius: "50%", background: "rgba(29,158,117,0.15)", display: "flex", alignItems: "center", justifyContent: "center", color: "#5DCAA5", fontSize: 16, flexShrink: 0 }}>✓</div>
          <div>
            <div style={{ fontSize: 13, color: "#5DCAA5", fontWeight: 500, marginBottom: 2 }}>Simülasyon başlatılmaya hazır</div>
            <div style={{ fontSize: 12, color: "rgba(240,244,248,0.55)" }}>
              {formData.devices.length} cihaz · {mode.icon} {mode.label} modu · 24 saatlik optimizasyon
            </div>
          </div>
        </div>

        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1rem" }}>
          <button onClick={() => goTo(formData.devices.some(d => d.apiName === "HVAC") ? "comfort" : "devices")} style={{ background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 20px", fontSize: 13, color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>← Geri</button>
          <button
            onClick={handleStart}
            disabled={loading}
            style={{ background: loading ? "rgba(29,158,117,0.4)" : "#1D9E75", border: "none", borderRadius: 8, padding: "11px 32px", fontSize: 14, fontWeight: 500, color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: loading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 10 }}>
            {loading ? (
              <>
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: "rgba(255,255,255,0.5)", animation: "blink 0.8s ease infinite" }} />
                Simülasyon çalışıyor...
              </>
            ) : (
              <>
                <div style={{ width: 7, height: 7, borderRadius: "50%", background: "rgba(255,255,255,0.5)", animation: "blink 1.5s ease infinite" }} />
                Simülasyonu Başlat
              </>
            )}
          </button>
        </div>
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
