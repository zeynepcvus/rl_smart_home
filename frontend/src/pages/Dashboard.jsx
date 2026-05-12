const priceColor = (p) => p <= 1.0 ? "rgba(29,158,117,0.6)" : p <= 1.8 ? "rgba(250,199,117,0.6)" : "rgba(226,75,74,0.6)";

export default function Dashboard({ goTo, formData, apiResult }) {
  const hours = apiResult?.rl?.hours || [];
  const summary = apiResult?.rl?.summary || {};
  const lastHour = hours[hours.length - 1] || {};
  const currentHour = hours.length;

  const hourlyPrices = hours.map(h => h.price);
  const hourlyCosts = hours.map(h => h.step_cost);
  const maxPrice = Math.max(...hourlyPrices, 1);
  const maxCost = Math.max(...hourlyCosts, 1);

  const indoorTemp = lastHour.indoor_temp ?? 22.0;
  const outdoorTemp = lastHour.outdoor_temp ?? 20.0;
  const currentPrice = lastHour.price ?? 0;
  const priceCategory = lastHour.price_category ?? "—";
  const activeDevices = lastHour.active_devices ?? [];

  const totalCost = summary.total_cost ?? 0;
  const comfortViolations = summary.comfort_violations ?? 0;

  const MODE_LABELS = {
    cost: { label: "Maliyet Odaklı", icon: "💰" },
    balanced: { label: "Dengeli", icon: "⚖️" },
    comfort: { label: "Konfor Odaklı", icon: "🌡️" },
  };
  const mode = MODE_LABELS[formData?.mode] || MODE_LABELS.balanced;
  const allDevices = ["HVAC", "Lighting", ...(formData?.devices?.map(d => d.name) || [])];

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", flexDirection: "column", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{ position: "fixed", inset: 0, backgroundImage: "linear-gradient(rgba(29,158,117,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.04) 1px, transparent 1px)", backgroundSize: "40px 40px", pointerEvents: "none" }} />

      <div style={{ display: "flex", alignItems: "center", padding: ".85rem 1.5rem", borderBottom: "0.5px solid rgba(255,255,255,0.07)", position: "relative", zIndex: 1, gap: "1rem" }}>
        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 14, color: "#5DCAA5", display: "flex", alignItems: "center", gap: 6, marginRight: "auto" }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#1D9E75", animation: "blink 2s ease infinite" }} />
          SmartHome RL
        </div>
        <span style={{ fontSize: 11, color: "#5DCAA5", padding: "3px 10px", borderRadius: 6, background: "rgba(29,158,117,0.12)", fontWeight: 500 }}>{mode.icon} {mode.label}</span>
        <span style={{ fontSize: 12, color: "#5DCAA5", padding: "5px 12px", borderRadius: 6, background: "rgba(29,158,117,0.12)", fontWeight: 500 }}>Dashboard</span>
        <span onClick={() => goTo("comparison")} style={{ fontSize: 12, color: "rgba(240,244,248,0.4)", padding: "5px 12px", borderRadius: 6, cursor: "pointer" }}>Karşılaştırma</span>
        <span style={{ fontSize: 12, color: "rgba(240,244,248,0.35)", marginLeft: "auto" }}>{apiResult ? `Simülasyon tamamlandı · ${currentHour} saat` : "Veri bekleniyor..."}</span>
      </div>

      <div style={{ flex: 1, padding: "1.25rem 1.5rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 10 }}>
        {!apiResult && (
          <div style={{ textAlign: "center", padding: "3rem", color: "rgba(240,244,248,0.35)", fontSize: 14 }}>
            Simülasyon verisi yok. Lütfen önce simülasyonu başlatın.
          </div>
        )}

        {apiResult && <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
            {[
              { label: "Günlük maliyet", val: `${totalCost} TL`, color: "#5DCAA5", sub: "24 saat tamamlandı" },
              { label: "Konfor ihlali", val: comfortViolations, color: comfortViolations > 3 ? "#FAC775" : "#5DCAA5", sub: "Bugün toplam" },
              { label: "Deadline ihlali", val: summary.deadline_violations ?? 0, color: "#5DCAA5", sub: "Bugün toplam" },
              { label: "HVAC anahtarı", val: summary.hvac_switches ?? 0, color: "#f0f4f8", sub: "Aç/kapat sayısı" },
            ].map(m => (
              <div key={m.label} style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".75rem 1rem" }}>
                <div style={{ fontSize: 10, color: "rgba(240,244,248,0.35)", letterSpacing: ".05em", textTransform: "uppercase", marginBottom: 4 }}>{m.label}</div>
                <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, color: m.color }}>{m.val}</div>
                <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginTop: 2 }}>{m.sub}</div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Son saat cihaz durumları</div>
              {allDevices.map(name => {
                const isActive = activeDevices.includes(name);
                return (
                  <div key={name} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", flexShrink: 0, background: isActive ? "#1D9E75" : "rgba(255,255,255,0.15)", boxShadow: isActive ? "0 0 4px rgba(29,158,117,0.6)" : "none" }} />
                    <span style={{ fontSize: 12, color: "rgba(240,244,248,0.8)", flex: 1 }}>{name}</span>
                    <span style={{ fontSize: 10, color: "rgba(240,244,248,0.35)" }}>{isActive ? "Aktif" : "Kapalı"}</span>
                  </div>
                );
              })}
            </div>

            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Son saat iç sıcaklık</div>
              <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 32, color: "#5DCAA5", marginBottom: 2 }}>{indoorTemp}°C</div>
              <div style={{ fontSize: 11, color: "rgba(240,244,248,0.35)", marginBottom: ".75rem" }}>Hedef bant: {formData?.tempMin ?? 20}–{formData?.tempMax ?? 24} °C</div>
              <div style={{ height: 6, background: "rgba(255,255,255,0.07)", borderRadius: 3, position: "relative", marginBottom: 4 }}>
                <div style={{ position: "absolute", height: "100%", background: "rgba(29,158,117,0.3)", borderRadius: 3, left: "28%", width: "29%" }} />
                <div style={{ position: "absolute", width: 3, height: 14, background: "#5DCAA5", borderRadius: 2, top: -4, left: `${Math.min(Math.max(((indoorTemp - 16) / 14) * 100, 0), 100)}%`, transform: "translateX(-50%)" }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(240,244,248,0.25)" }}>
                <span>16°C</span><span>30°C</span>
              </div>
              <div style={{ marginTop: ".6rem", fontSize: 11, color: "rgba(240,244,248,0.35)" }}>Dış sıcaklık: {outdoorTemp}°C</div>
            </div>

            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Elektrik fiyatı profili</div>
              <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, color: "#FAC775", marginBottom: 2 }}>{currentPrice} TL/kWh</div>
              <div style={{ fontSize: 10, color: "rgba(250,199,117,0.6)", marginBottom: ".6rem" }}>{priceCategory}</div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 52, marginBottom: 3 }}>
                {hourlyPrices.map((p, i) => (
                  <div key={i} style={{ flex: 1, borderRadius: "2px 2px 0 0", background: priceColor(p), height: `${(p / maxPrice) * 100}%` }} />
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(240,244,248,0.2)" }}>
                <span>00</span><span>06</span><span>12</span><span>18</span><span>23</span>
              </div>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 10 }}>
            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Saatlik maliyet</div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 40 }}>
                {hourlyCosts.map((c, i) => (
                  <div key={i} style={{ flex: 1, borderRadius: "2px 2px 0 0", background: priceColor(hourlyPrices[i]), height: `${Math.max((c / maxCost) * 100, 5)}%` }} />
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(240,244,248,0.2)", marginTop: 3 }}>
                <span>00:00</span><span>06:00</span><span>12:00</span><span>18:00</span><span>23:00</span>
              </div>
            </div>

            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Konfor skoru</div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                {(() => {
                  const score = Math.max(0, Math.round((1 - comfortViolations / 24) * 100));
                  const dash = 138.2;
                  const offset = dash - (dash * score / 100);
                  return (
                    <>
                      <svg width="52" height="52" viewBox="0 0 52 52">
                        <circle cx="26" cy="26" r="22" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="4"/>
                        <circle cx="26" cy="26" r="22" fill="none" stroke="#1D9E75" strokeWidth="4" strokeDasharray={dash} strokeDashoffset={offset} strokeLinecap="round" transform="rotate(-90 26 26)"/>
                        <text x="26" y="31" textAnchor="middle" fontSize="13" fontWeight="500" fill="#5DCAA5" fontFamily="DM Serif Display,serif">{score}%</text>
                      </svg>
                      <div>
                        <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 20, color: "#5DCAA5" }}>{score >= 80 ? "İyi" : score >= 60 ? "Orta" : "Düşük"}</div>
                        <div style={{ fontSize: 10, color: "rgba(240,244,248,0.35)", marginTop: 1 }}>Konfor puanı</div>
                        <div style={{ fontSize: 10, color: "rgba(240,244,248,0.35)", marginTop: 3 }}>{comfortViolations} ihlal bugün</div>
                      </div>
                    </>
                  );
                })()}
              </div>
              <button onClick={() => goTo("comparison")} style={{ width: "100%", marginTop: 10, background: "rgba(29,158,117,0.1)", border: "0.5px solid rgba(29,158,117,0.3)", borderRadius: 8, padding: 10, fontSize: 13, fontWeight: 500, color: "#5DCAA5", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" }}>
                RL vs Kural Tabanlı →
              </button>
            </div>
          </div>
        </>}
      </div>
      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
    </div>
  );
}
