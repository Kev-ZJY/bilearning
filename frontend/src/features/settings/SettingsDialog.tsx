import { useEffect, useRef, useState } from "react";
import * as api from "../../api/client";
import { isDesktop } from "../../api/desktop";
import type { Settings } from "../../api/types";

/**
 * 表单态。**不复用 `Settings`**，因为它多了一个 `""` 分支。
 *
 * 正在编辑的数值框可以是空的（用户按了退格），而 `Settings.llm_max_tokens`
 * 是 `number` —— 直接用 `Settings` 当表单类型，就只能靠 `as` 硬骗类型
 * 编译器，骗来的类型会一路传染到提交逻辑，最后变成「空值发出去 → 后端 422」。
 * 把「空」显式建模成合法状态，才能在提交前拦下来。
 */
type Numeric = number | "";
type BusyAction = "saving" | "testing" | "resetting" | "session-profile" | null;

interface FormState {
  llm_base_url: string;
  llm_model: string;
  llm_max_tokens: Numeric;
  llm_temperature: Numeric;
  llm_timeout: Numeric;
}

export function SettingsDialog({
  open,
  onClose,
  onNotice,
}: {
  open: boolean;
  onClose: () => void;
  onNotice: (msg: string | null) => void;
}) {
  const [form, setForm] = useState<FormState | null>(null);
  // 掩码与「有没有配」单独存：它们只用于渲染占位符，
  // 混进 form 每次渲染都要判「这个值是掩码还是新值」。
  const [keyInfo, setKeyInfo] = useState<{masked: string; set: Settings["llm_api_key_set"]}>({ masked: "", set: null });
  // apiKey 初始为空（只放掩码占位），用户不填就保持空 = 不改。
  const [apiKey, setApiKey] = useState("");
  const [busyAction, setBusyAction] = useState<BusyAction>(null);
  const busyActionRef = useRef<BusyAction>(null);
  const [testResult,setTestResult]=useState('');
  const [loadError, setLoadError] = useState<string | null>(null);

  const beginAction = (action: Exclude<BusyAction, null>) => {
    if (busyActionRef.current !== null) return false;
    busyActionRef.current = action;
    setBusyAction(action);
    return true;
  };
  const endAction = () => {
    busyActionRef.current = null;
    setBusyAction(null);
  };

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoadError(null);
    setTestResult('');
    api
      .getSettings()
      .then((s: Settings) => {
        if (!alive) return;
        setForm({
          llm_base_url: s.llm_base_url,
          llm_model: s.llm_model,
          llm_max_tokens: s.llm_max_tokens,
          llm_temperature: s.llm_temperature,
          llm_timeout: s.llm_timeout,
        });
        setKeyInfo({ masked: s.llm_api_key_masked, set: s.llm_api_key_set });
        setApiKey("");
      })
      .catch((e) => {
        if (!alive) return;
        setLoadError(
          e instanceof Error ? e.message : String(e)
        );
      });
    return () => {
      alive = false;
    };
  }, [open]);

  if (!open) return null;

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((cur) => (cur ? { ...cur, [k]: v } : cur));

  /** 读一个数值框。空串保持空（这是用户的真实状态），其余转数字。 */
  const setNumeric = (k: "llm_max_tokens" | "llm_temperature" | "llm_timeout", raw: string) =>
    set(k, (raw.trim() === "" ? "" : Number(raw)) as FormState[typeof k]);

  // 提交前的自检。返回出错信息，null 表示通过。
  //
  // 为什么必须在前端拦：`Number("")` 是 0，把 0 发给后端必然 422，
  // 而 pydantic 的 "llm_max_tokens: greater than or equal to 1" 对
  // 一个刚退格了一下的用户毫无意义。空值不是「后端拒绝」，是「还没填完」。
  const validate = (f: FormState): string | null => {
    if (!f.llm_model.trim()) return "模型名不能为空";
    if (!f.llm_base_url.trim()) return "接口地址不能为空";
    if (f.llm_max_tokens === "" || !Number.isFinite(f.llm_max_tokens) || f.llm_max_tokens < 1)
      return "输出预算 max_tokens 必须是大于 0 的整数";
    if (f.llm_timeout === "" || !Number.isFinite(f.llm_timeout) || f.llm_timeout < 1)
      return "超时（秒）必须是大于 0 的整数";
    if (f.llm_temperature === "" || !Number.isFinite(f.llm_temperature))
      return "temperature 必须是一个数字";
    return null;
  };

  const save = async () => {
    if (!form) return;
    // 注意这里**没有**再包一层 Number()。转换只发生在 onChange 一处：
    // 两处都转的话，onChange 的转换就变得无法被任何测试观察到 ——
    // 守卫写在错误的层，等于没有守卫（本项目实测踩过，见 test 里的注释）。
    const bad = validate(form);
    if (bad) {
      onNotice(`无法保存：${bad}`);
      return;
    }
    if (!beginAction("saving")) return;
    try {
      const payload: Record<string, unknown> = {
        llm_base_url: form.llm_base_url.trim(),
        llm_model: form.llm_model.trim(),
        llm_max_tokens: form.llm_max_tokens as number,
        llm_temperature: form.llm_temperature as number,
        llm_timeout: form.llm_timeout as number,
      };
      // 只在用户真的输入了新密钥时才提交这一项。
      // 提交空串会被后端当成「把密钥设成空」——真实事故。
      if (apiKey.trim()) payload.llm_api_key = apiKey.trim();
      await api.updateSettings(payload);
      setApiKey("");
      onNotice("设置已保存，立即生效");
      onClose();
    } catch (e) {
      // 刻意不清空表单、不关窗：用户填的地址和密钥必须留在屏幕上，
      // 否则他要重新手打一遍，而密钥他根本看不到。
      onNotice(`保存失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      endAction();
    }
  };

  const testConnection = async () => {
    if(!form)return;
    const bad=validate(form);if(bad){setTestResult(bad);return}
    if (!beginAction("testing")) return;
    setTestResult('连接中…');
    try { const result=await api.testModelConnection({...form,...(apiKey.trim()?{llm_api_key:apiKey.trim()}:{})});setTestResult(`连接成功 · ${(result.elapsed_ms/1000).toFixed(1)} 秒`); }
    catch(e){setTestResult(e instanceof Error?e.message:String(e))} finally{endAction()}
  };

  const useSessionProfile = async () => {
    if (!isDesktop()) {
      onNotice("仅桌面应用可使用本次运行配置");
      return;
    }
    if (!form) return;
    const bad = validate(form);
    if (bad) {
      onNotice(`无法应用本次运行配置：${bad}`);
      return;
    }
    const sessionKey = apiKey.trim();
    if (!sessionKey) {
      onNotice("请输入非空密钥");
      return;
    }
    if (!beginAction("session-profile")) return;
    try {
      await api.setSessionProfile({
        llm_base_url: form.llm_base_url.trim(),
        llm_model: form.llm_model.trim(),
        llm_api_key: sessionKey,
        llm_max_tokens: form.llm_max_tokens as number,
        llm_temperature: form.llm_temperature as number,
        llm_timeout: form.llm_timeout as number,
      });
      setApiKey("");
      onNotice("配置仅本次运行使用；重启后恢复已保存配置。");
      onClose();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      onNotice(
        `本次运行配置设置失败：${message.includes(sessionKey) ? "错误详情已隐藏" : message}`
      );
    } finally {
      endAction();
    }
  };

  const resetToEnv = async () => {
    if (!beginAction("resetting")) return;
    try {
      // 「恢复默认」= 把这些行的覆盖删掉，回到环境变量。
      // 不是把字段置空——置空会让请求发不出去（模型名不能为空）。
      const s = await api.updateSettings({
        llm_base_url: null,
        llm_model: null,
        llm_max_tokens: null,
        llm_temperature: null,
        llm_timeout: null,
      });
      setForm({
        llm_base_url: s.llm_base_url,
        llm_model: s.llm_model,
        llm_max_tokens: s.llm_max_tokens,
        llm_temperature: s.llm_temperature,
        llm_timeout: s.llm_timeout,
      });
      onNotice("已恢复为环境变量里的默认值");
    } catch (e) {
      onNotice(`恢复失败：${e instanceof Error ? e.message : String(e)}`);
    } finally {
      endAction();
    }
  };

  return (
    <div
      className="modal-mask"
      onClick={(e) => {
        // 点遮罩关闭，点内容不关（否则用户想点输入框却把窗关了）
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal" role="dialog" aria-label="模型设置">
        <header className="modal__header">
          <span>模型设置</span>
          <button className="modal__close" onClick={onClose} aria-label="关闭">
            ✕
          </button>
        </header>

        {loadError ? (
          <div className="modal__body">
            <p className="settings__error">读取设置失败：{loadError}</p>
            <p className="settings__hint">
              后端可能没启动，或用的是旧版本（没有 /api/settings 端点）。
            </p>
          </div>
        ) : !form ? (
          <div className="modal__body">读取中…</div>
        ) : (
          <div className="modal__body">
            <p className="settings__hint">
              支持 OpenAI 兼容接口。保存后用于新的对话与视频整理。
            </p>

            <label className="settings__field">
              <span>接口地址</span>
              <input
                value={form.llm_base_url}
                placeholder="https://api.example.com/v1/chat/completions"
                onChange={(e) => set("llm_base_url", e.target.value)}
              />
            </label>

            <label className="settings__field">
              <span>模型</span>
              <input
                value={form.llm_model}
                placeholder="模型名"
                onChange={(e) => set("llm_model", e.target.value)}
              />
            </label>

            <label className="settings__field">
              <span>API Key</span>
              <input
                type="password"
                value={apiKey}
                placeholder={
                  keyInfo.set === null
                    ? "密钥状态未验证 · 留空保留原密钥"
                    : keyInfo.set
                    ? `已配置（尾号 ${keyInfo.masked.split(":").pop()}）· 留空保持不变`
                    : "尚未配置"
                }
                onChange={(e) => setApiKey(e.target.value)}
              />
            </label>

            <div className="settings-test"><span role="status">{testResult}</span><button className="btn-new" onClick={testConnection} disabled={busyAction !== null}>{busyAction === "testing"?'测试中…':'测试连接'}</button></div>
            {isDesktop() && (
              <div className="settings-test">
                <span className="settings__hint">重启后失效，保留已保存设置</span>
                <button
                  className="btn-new"
                  onClick={useSessionProfile}
                  disabled={busyAction !== null}
                >
                  {busyAction === "session-profile" ? "设置中…" : "仅本次运行使用配置"}
                </button>
              </div>
            )}
            <details className="settings-advanced"><summary>高级配置</summary>
            <div className="settings__row">
              <label className="settings__field">
                <span>输出预算 max_tokens</span>
                <input
                  type="number"
                  value={form.llm_max_tokens}
                  onChange={(e) => setNumeric("llm_max_tokens", e.target.value)}
                />
              </label>
              <label className="settings__field">
                <span>超时（秒）</span>
                <input
                  type="number"
                  value={form.llm_timeout}
                  onChange={(e) => setNumeric("llm_timeout", e.target.value)}
                />
              </label>
            </div>

            <label className="settings__field">
              <span>temperature</span>
                <input
                  type="number"
                  step="0.1"
                  value={form.llm_temperature}
                  onChange={(e) => setNumeric("llm_temperature", e.target.value)}
                />
              </label>

            <p className="settings__hint">
              输出预算应按所选模型支持的上限调整；部分模型会将思考内容计入预算。
              预算不足可能导致输出截断，过高也可能被接口拒绝。
            </p>

            </details>
            <div className="modal__actions">
              <button
                className="btn-new"
                onClick={resetToEnv}
                disabled={busyAction !== null}
              >
                恢复默认
              </button>
              <button
                className="btn-new btn-new--primary"
                onClick={save}
                disabled={busyAction !== null}
              >
                {busyAction === "saving" ? "保存中…" : "保存"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
