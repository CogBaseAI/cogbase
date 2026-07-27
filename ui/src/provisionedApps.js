// Language sync for the auto-provisioned contract-analyst app.
//
// The provisioner (api/provisioning.py) seeds every new account with a
// contract-analyst app whose config ships single-language (English). App config
// intentionally stays single-language — instead, when the UI language changes we
// PATCH the app's UI-only presentation fields (query_intro, example_queries) via
// the light endpoint PATCH /namespaces/{ns}/applications/{name}/config
// (api/routers/applications.py), so the Query tab's starter panel reads in the
// active language without a rebuild.

// The provisioner always names the app this.
export const PROVISIONED_APP_NAME = 'contract-analyst'

// One preset per supported UI language. The `en` entry mirrors
// examples/contract_analyst_demo/config.yaml verbatim so an untouched
// provisioned app is recognized as "still default" and safe to re-localize; a
// user's own edits (matching no preset) are left alone.
export const PROVISIONED_PRESETS = {
  en: {
    query_intro:
      'Ask across the contract portfolio, or have a single contract reviewed clause by clause and returned as a tracked-changes redline.',
    example_queries: [
      'review an uploaded contract',
      'which of my contracts expire in the next 90 days?',
      'which agreements have unlimited liability or no stated liability cap?',
      'flag any auto-renewal or unusual termination clauses across my contracts',
    ],
  },
  zh: {
    query_intro:
      '在整个合同组合中提问，或让某一份合同逐条审阅，并以修订标记（redline）形式返回。',
    example_queries: [
      '审阅一份已上传的合同',
      '我的哪些合同将在未来 90 天内到期？',
      '哪些协议存在无限责任或未约定责任上限？',
      '标出我的合同中任何自动续约或异常的终止条款',
    ],
  },
}

const sameQueries = (a, b) => JSON.stringify(a || []) === JSON.stringify(b || [])

const matchesPreset = (intro, queries, preset) =>
  intro === preset.query_intro && sameQueries(queries, preset.example_queries)

// Re-localize the provisioned app's starter panel to `lang`. Best-effort and
// idempotent: a no-op when the app is absent, the language has no preset, the
// content is already in `lang`, or the user has customized the fields.
export async function syncProvisionedAppLanguage({ apps, lang, apiUrl, authFetch, refreshApps }) {
  const app = (apps || []).find(a => a.name === PROVISIONED_APP_NAME)
  const desired = PROVISIONED_PRESETS[lang]
  if (!app || !desired) return

  const cfg = app.config || {}
  const curIntro = typeof cfg.query_intro === 'string' ? cfg.query_intro : ''
  const curQueries = Array.isArray(cfg.example_queries) ? cfg.example_queries : []

  if (matchesPreset(curIntro, curQueries, desired)) return   // already localized

  // Only overwrite content that still matches some known preset (untouched
  // provisioned defaults); leave a user's own edits untouched.
  const isDefault = Object.values(PROVISIONED_PRESETS).some(p =>
    matchesPreset(curIntro, curQueries, p))
  if (!isDefault) return

  const url = `${apiUrl}/namespaces/${encodeURIComponent(app.namespace)}/applications/${encodeURIComponent(app.name)}/config`
  try {
    const resp = await authFetch(url, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query_intro: desired.query_intro,
        example_queries: desired.example_queries,
      }),
    })
    if (resp.ok) await refreshApps()
  } catch {
    // Best-effort — leave config as-is on failure.
  }
}
