import Sidebar from "../components/Sidebar";

const TEMP_MIN_RECOMMENDED = 20;
const TEMP_MAX_RECOMMENDED = 24;
const TEMP_MIN_LOW = 18;
const TEMP_MIN_HIGH = 22;
const TEMP_MAX_LOW = 22;
const TEMP_MAX_HIGH = 26;

function getTempWarning(key, value, formData) {
  if (key === "tempMin") {
    if (value < TEMP_MIN_LOW) return `Çok düşük — önerilen minimum ${TEMP_MIN_RECOMMENDED}°C`;
    if (value > TEMP_MIN_HIGH) return `Çok yüksek — önerilen minimum ${TEMP_MIN_RECOMMENDED}°C`;
  }
  if (key === "tempMax") {
    if (value < TEMP_MAX_LOW) return `Çok düşük — önerilen maksimum ${TEMP_MAX_RECOMMENDED}°C`;
    if (value > TEMP_MAX_HIGH) return `Çok yüksek — önerilen maksimum ${TEMP_MAX_RECOMMENDED}°C`;
  }
  if ((formData.tempMax - formData.tempMin) < 2) return "Aralık çok dar — HVAC sürekli çalışabilir";
  return null;
}

export default function ComfortPrefs({ goTo, formData, updateForm }) {
  const handleNext = () => {
    goTo("summary");
  };

  const fields = [
    { label: "Minimum sıcaklık", key: "tempMin", icon: "🥶" },
    { label: "Maksimum sıcaklık", key: "tempMax", icon: "🥵" },
  ];

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

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: "1.25rem" }}>
            {fields.map(field => {
              const warning = getTempWarning(field.key, formData[field.key], formData);
              return (
                <div key={field.key} style={{
                  padding: "12px 14px",
                  background: "rgba(255,255,255,0.03)",
                  borderRadius: 10,
                  border: warning ? "0.5px solid rgba(239,159,39,0.4)" : "0.5px solid rgba(255,255,255,0.08)",
                  transition: "border .2s"
                }}>
                  <div style={{ fontSize: 11, color: "rgba(240,244,248,0.45)", marginBottom: 8, display: "flex", alignItems: "center", gap: 5 }}>
                    <span>{field.icon}</span>
                    <span style={{ textTransform: "uppercase", letterSpacing: ".06em", fontWeight: 500 }}>{field.label}</span>
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontFamily: "'DM Serif Display', serif", fontSize: 28, fontWeight: 700, color: warning ? "#EF9F27" : "#FAC775", transition: "color .2s" }}>
                      {formData[field.key]}°C
                    </span>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <button
                        onClick={() => {
                          const val = formData[field.key] + 1;
                          if (field.key === "tempMin" && val >= formData.tempMax) return;
                          if (field.key === "tempMax" && val > 30) return;
                          updateForm({ [field.key]: val });
                        }}
                        style={{ width: 28, height: 22, borderRadius: 5, border: "0.5px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.06)", color: "#f0f4f8", cursor: "pointer", fontSize: 13, lineHeight: 1 }}>▲</button>
                      <button
                        onClick={() => {
                          const val = formData[field.key] - 1;
                          if (field.key === "tempMax" && val <= formData.tempMin) return;
                          if (field.key === "tempMin" && val < 16) return;
                          updateForm({ [field.key]: val });
                        }}
                        style={{ width: 28, height: 22, borderRadius: 5, border: "0.5px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.06)", color: "#f0f4f8", cursor: "pointer", fontSize: 13, lineHeight: 1 }}>▼</button>
                    </div>
                  </div>

                  <div style={{ fontSize: 10, color: "rgba(240,244,248,0.25)", marginTop: 6 }}>
                    16°C – 30°C arası
                  </div>

                  {warning && (
                    <div style={{
                      marginTop: 8,
                      padding: "5px 8px",
                      background: "rgba(239,159,39,0.1)",
                      border: "0.5px solid rgba(239,159,39,0.35)",
                      borderRadius: 6,
                      fontSize: 10,
                      color: "#EF9F27",
                      display: "flex",
                      alignItems: "center",
                      gap: 5,
                    }}>
                      <span>⚠</span>
                      <span>{warning}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

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
          <button onClick={() => goTo("devices")} style={{ background: "rgba(255,255,255,0.05)", border: "0.5px solid rgba(255,255,255,0.2)", borderRadius: 8, padding: "10px 20px", fontSize: 13, color: "rgba(240,244,248,0.7)", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>← Geri</button>
          <button onClick={handleNext} style={{ background: "#1D9E75", border: "none", borderRadius: 8, padding: "10px 28px", fontSize: 13, fontWeight: 500, color: "#fff", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>Devam Et →</button>
        </div>
      </div>
      <style>{`select option { background: #0a1628; }`}</style>
    </div>
  );
}
