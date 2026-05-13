import Sidebar from "../components/Sidebar";

export default function ComfortPrefs({ goTo, formData, updateForm }) {
  const handleNext = () => {
    goTo("summary");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{
        position: "fixed", inset: 0,
        backgroundImage: "linear-gradient(rgba(29,158,117,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.05) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none"
      }} />

      <Sidebar currentStep="comfort" />

      <div style={{ flex: 1, padding: "2rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column" }}>

        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 26, color: "#f0f4f8", marginBottom: ".35rem" }}>
          Konfor tercihlerini belirle
        </div>
        <p style={{ fontSize: 13, color: "rgba(240,244,248,0.65)", marginBottom: "1.75rem", fontWeight: 300 }}>
          HVAC sistemi bu sınırlar içinde iç sıcaklığı koruyacak şekilde çalışır.
        </p>

        <div style={{ background: "rgba(255,255,255,0.04)", border: "0.5px solid rgba(255,255,255,0.1)", borderLeft: "3px solid #FAC775", borderRadius: 12, padding: "1.25rem", marginBottom: "1rem" }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "#FAC775", letterSpacing: ".07em", textTransform: "uppercase", marginBottom: "1.25rem" }}>
            🌡️ Sıcaklık bandı
          </div>
          {[
            { label: "Minimum sıcaklık", key: "tempMin", min: 16, max: 22 },
            { label: "Maksimum sıcaklık", key: "tempMax", min: 22, max: 30 },
          ].map(field => (
            <div key={field.key} style={{ marginBottom: "1.25rem", padding: "10px 12px", background: "rgba(255,255,255,0.03)", borderRadius: 9, border: "0.5px solid rgba(255,255,255,0.07)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: ".7rem" }}>
                <span style={{ fontSize: 13, color: "rgba(240,244,248,0.75)", fontWeight: 500 }}>{field.label}</span>
                <span style={{ fontSize: 17, fontWeight: 700, color: "#FAC775", fontFamily: "'DM Serif Display', serif" }}>{formData[field.key]} °C</span>
              </div>
              <input
                type="range" min={field.min} max={field.max} value={formData[field.key]}
                onChange={e => {
                  const val = parseInt(e.target.value);
                  if (field.key === "tempMin" && val >= formData.tempMax) return;
                  if (field.key === "tempMax" && val <= formData.tempMin) return;
                  updateForm({ [field.key]: val });
                }}
                style={{ width: "100%", accentColor: "#1D9E75", cursor: "pointer" }}
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "rgba(240,244,248,0.3)", marginTop: 4 }}>
                <span>{field.min} °C</span><span>{field.max} °C</span>
              </div>
            </div>
          ))}

          <div style={{ display: "flex", alignItems: "center", gap: "1rem", padding: ".85rem 1rem", background: "rgba(250,199,117,0.07)", border: "0.5px solid rgba(250,199,117,0.25)", borderRadius: 9, marginTop: ".5rem" }}>
            <span style={{ fontSize: 12, color: "rgba(240,244,248,0.45)", whiteSpace: "nowrap" }}>16°C</span>
            <div style={{ flex: 1, height: 8, background: "rgba(255,255,255,0.07)", borderRadius: 4, position: "relative", overflow: "hidden" }}>
              <div style={{
                position: "absolute", height: "100%",
                background: "linear-gradient(90deg, #85B7EB, #FAC775, #EF9F27)",
                borderRadius: 4,
                left: `${((formData.tempMin - 16) / 14) * 100}%`,
                width: `${((formData.tempMax - formData.tempMin) / 14) * 100}%`,
                transition: "left .3s, width .3s"
              }} />
            </div>
            <span style={{ fontSize: 12, color: "rgba(240,244,248,0.45)", whiteSpace: "nowrap" }}>30°C</span>
            <span style={{ fontSize: 12, color: "#FAC775", fontWeight: 600, whiteSpace: "nowrap" }}>
              {formData.tempMin}° – {formData.tempMax}°C
            </span>
          </div>
        </div>


        <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", paddingTop: "1.25rem" }}>
          <button onClick={() => goTo("devices")} style={{ background: "transparent", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 8, padding: "10px 20px", fontSize: 13, color: "rgba(240,244,248,0.45)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>← Geri</button>
          <button onClick={handleNext} style={{ background: "#1D9E75", border: "none", borderRadius: 8, padding: "10px 28px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>Devam Et →</button>
        </div>
      </div>
      <style>{`select option { background: #0a1628; }`}</style>
    </div>
  );
}