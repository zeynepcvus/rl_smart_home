const priceColor = (p) => p <= 1.0 ? "rgba(29,158,117,0.85)" : p <= 1.8 ? "rgba(250,199,117,0.85)" : "rgba(226,75,74,0.85)";

export default function Dashboard({ goTo, formData, apiResult }) {
  const hours = apiResult?.rl?.hours || [];
  const summary = apiResult?.rl?.summary || {};
  const lastHour = hours[hours.length - 1] || {};


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
  const allDeviceEntries = [
    { display: "HVAC", api: "HVAC" },
    { display: "Lighting", api: "Lighting" },
    ...(formData?.devices
      ?.filter(d => d.apiName !== "HVAC" && d.apiName !== "Lighting")
      ?.map(d => ({ display: d.name, api: d.apiName ?? d.name })) || []),
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#0a1628", display: "flex", flexDirection: "column", fontFamily: "'DM Sans', sans-serif" }}>
      <div style={{ position: "fixed", inset: 0, backgroundImage: "linear-gradient(rgba(29,158,117,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(29,158,117,0.04) 1px, transparent 1px)", backgroundSize: "40px 40px", pointerEvents: "none" }} />

      <div style={{ display: "flex", alignItems: "center", padding: ".85rem 1.5rem", borderBottom: "0.5px solid rgba(255,255,255,0.07)", position: "relative", zIndex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, width: 160, flexShrink: 0 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#1D9E75", boxShadow: "0 0 6px rgba(29,158,117,0.8)", animation: "blink 2s ease infinite", flexShrink: 0 }} />
          <span style={{ fontFamily: "'DM Serif Display', serif", fontSize: 15, color: "#f0f4f8" }}>SmartHome</span>
          <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 11, fontWeight: 700, color: "#1D9E75", background: "rgba(29,158,117,0.15)", border: "0.5px solid rgba(29,158,117,0.4)", borderRadius: 5, padding: "1px 6px", letterSpacing: ".04em" }}>RL</span>
        </div>
        <div style={{ flex: 1, display: "flex", justifyContent: "center", gap: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#f0f4f8", padding: "6px 16px", borderRadius: 8, background: "rgba(29,158,117,0.15)", border: "0.5px solid rgba(29,158,117,0.4)" }}>Dashboard</span>
          <span onClick={() => goTo("comparison")} style={{ fontSize: 13, color: "rgba(240,244,248,0.45)", padding: "6px 16px", borderRadius: 8, cursor: "pointer", border: "0.5px solid transparent" }}>Karşılaştırma</span>
        </div>
        <div style={{ width: 160, flexShrink: 0, display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 11, color: "#5DCAA5", padding: "3px 10px", borderRadius: 6, background: "rgba(29,158,117,0.1)", fontWeight: 500 }}>{mode.icon} {mode.label}</span>
          <span onClick={() => goTo("welcome")} style={{ fontSize: 12, color: "rgba(240,244,248,0.65)", cursor: "pointer", padding: "5px 10px", borderRadius: 6, border: "0.5px solid rgba(255,255,255,0.2)", background: "rgba(255,255,255,0.04)", transition: "all .15s" }}>↩ Başa dön</span>
        </div>
      </div>

      <div style={{ flex: 1, padding: "1.25rem 1.5rem", position: "relative", zIndex: 1, display: "flex", flexDirection: "column", gap: 10 }}>
        {!apiResult && (
          <div style={{ textAlign: "center", padding: "3rem", color: "rgba(240,244,248,0.35)", fontSize: 14 }}>
            Simülasyon verisi yok. Lütfen önce simülasyonu başlatın.
          </div>
        )}

        {apiResult && <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
            {(() => {
              const dl = summary.deadline_violations ?? 0;
              const hvac = summary.hvac_switches ?? 0;
              const comfortAccent = comfortViolations === 0 ? "#1D9E75" : comfortViolations <= 3 ? "#FAC775" : "#e24b4a";
              const deadlineAccent = dl === 0 ? "#1D9E75" : dl <= 2 ? "#FAC775" : "#e24b4a";
              return [
                { label: "Günlük maliyet", val: `${totalCost} TL`, color: "#5DCAA5", accent: "#1D9E75", sub: "24 saat tamamlandı" },
                { label: "Konfor ihlali", val: comfortViolations, color: comfortAccent, accent: comfortAccent, sub: "Bugün toplam" },
                { label: "Deadline ihlali", val: dl, color: deadlineAccent, accent: deadlineAccent, sub: "Bugün toplam" },
                { label: "HVAC anahtarı", val: hvac, color: "#85B7EB", accent: "#85B7EB", sub: "Aç/kapat sayısı" },
              ].map(m => (
                <div key={m.label} style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "0.5px solid rgba(255,255,255,0.08)",
                  borderLeft: `3px solid ${m.accent}`,
                  borderRadius: 10,
                  padding: ".75rem 1rem",
                  boxShadow: `inset 0 0 20px rgba(0,0,0,0.1)`,
                }}>
                  <div style={{ fontSize: 10, color: "rgba(240,244,248,0.35)", letterSpacing: ".05em", textTransform: "uppercase", marginBottom: 4 }}>{m.label}</div>
                  <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 22, color: m.color }}>{m.val}</div>
                  <div style={{ fontSize: 10, color: "rgba(240,244,248,0.3)", marginTop: 2 }}>{m.sub}</div>
                </div>
              ));
            })()}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Simülasyon Sonu · Cihaz Durumları</div>
              {allDeviceEntries.map(entry => {
                const isActive = activeDevices.includes(entry.api);
                return (
                  <div key={entry.api} style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 0", borderBottom: "0.5px solid rgba(255,255,255,0.05)" }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", flexShrink: 0, background: isActive ? "#1D9E75" : "rgba(255,255,255,0.15)", boxShadow: isActive ? "0 0 4px rgba(29,158,117,0.6)" : "none" }} />
                    <span style={{ fontSize: 12, color: "rgba(240,244,248,0.8)", flex: 1 }}>{entry.display}</span>
                    <span style={{ fontSize: 10, color: isActive ? "#5DCAA5" : "rgba(240,244,248,0.35)" }}>{isActive ? "Aktif" : "Kapalı"}</span>
                  </div>
                );
              })}
            </div>

            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase", marginBottom: ".75rem" }}>Son saat iç sıcaklık</div>
              <div style={{ fontFamily: "'DM Serif Display', serif", fontSize: 32, color: "#5DCAA5", marginBottom: 2 }}>{indoorTemp}°C</div>
              <div style={{ fontSize: 11, color: "rgba(240,244,248,0.35)", marginBottom: ".75rem" }}>Hedef bant: {formData?.tempMin ?? 20}–{formData?.tempMax ?? 24} °C</div>
              <div style={{ height: 6, background: "rgba(255,255,255,0.07)", borderRadius: 3, position: "relative", marginBottom: 4 }}>
                <div style={{ position: "absolute", height: "100%", background: "rgba(29,158,117,0.3)", borderRadius: 3, left: `${(((formData?.tempMin ?? 20) - 16) / 14) * 100}%`, width: `${(((formData?.tempMax ?? 24) - (formData?.tempMin ?? 20)) / 14) * 100}%` }} />
                <div style={{ position: "absolute", width: 3, height: 14, background: "#5DCAA5", borderRadius: 2, top: -4, left: `${Math.min(Math.max(((indoorTemp - 16) / 14) * 100, 0), 100)}%`, transform: "translateX(-50%)" }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "rgba(240,244,248,0.25)" }}>
                <span>16°C</span><span>30°C</span>
              </div>
              <div style={{ marginTop: ".6rem", fontSize: 11, color: "rgba(240,244,248,0.35)" }}>Dış sıcaklık: {outdoorTemp}°C</div>
            </div>

            <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: ".85rem 1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: ".75rem" }}>
                <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase" }}>Elektrik fiyatı profili</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {[{ color: "rgba(29,158,117,0.85)", label: "Ucuz" }, { color: "rgba(250,199,117,0.85)", label: "Orta" }, { color: "rgba(226,75,74,0.85)", label: "Pahalı" }].map(l => (
                    <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                      <div style={{ width: 7, height: 7, borderRadius: 2, background: l.color }} />
                      <span style={{ fontSize: 9, color: "rgba(240,244,248,0.3)" }}>{l.label}</span>
                    </div>
                  ))}
                </div>
              </div>
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
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: ".75rem" }}>
                <div style={{ fontSize: 10, fontWeight: 500, color: "rgba(240,244,248,0.35)", letterSpacing: ".06em", textTransform: "uppercase" }}>Saatlik maliyet</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {[{ color: "rgba(29,158,117,0.85)", label: "Ucuz" }, { color: "rgba(250,199,117,0.85)", label: "Orta" }, { color: "rgba(226,75,74,0.85)", label: "Pahalı" }].map(l => (
                    <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                      <div style={{ width: 7, height: 7, borderRadius: 2, background: l.color }} />
                      <span style={{ fontSize: 9, color: "rgba(240,244,248,0.3)" }}>{l.label}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 80 }}>
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
