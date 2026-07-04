import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fmtBytes, fmtRelTime, previewText, metaText, schemaTypeStr, simplifyExtractionSchemas, streamSSE, waitForTasks, resolveArtifactLinks, artifactLabel } from '../utils'

describe('fmtBytes', () => {
  it('formats bytes', () => expect(fmtBytes(512)).toBe('512 B'))
  it('formats kilobytes', () => expect(fmtBytes(2048)).toBe('2.0 KB'))
  it('formats megabytes', () => expect(fmtBytes(2 * 1024 * 1024)).toBe('2.0 MB'))
})

describe('fmtRelTime', () => {
  const NOW = new Date('2026-07-03T12:00:00Z').getTime()
  beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(NOW) })
  const ago = (ms) => new Date(NOW - ms).toISOString()

  it('returns empty for missing/invalid input', () => {
    expect(fmtRelTime()).toBe('')
    expect(fmtRelTime('not a date')).toBe('')
  })
  it('says "just now" under 45s', () => expect(fmtRelTime(ago(10 * 1000))).toBe('just now'))
  it('formats minutes', () => expect(fmtRelTime(ago(5 * 60 * 1000))).toBe('5m'))
  it('formats hours', () => expect(fmtRelTime(ago(3 * 60 * 60 * 1000))).toBe('3h'))
  it('formats days', () => expect(fmtRelTime(ago(2 * 24 * 60 * 60 * 1000))).toBe('2d'))
  it('falls back to a locale date beyond a week', () => {
    const out = fmtRelTime(ago(30 * 24 * 60 * 60 * 1000))
    expect(out).not.toMatch(/^(just now|\d+[mhd])$/)
    expect(out).toBe(new Date(NOW - 30 * 24 * 60 * 60 * 1000).toLocaleDateString())
  })
})

describe('previewText', () => {
  it('returns short text unchanged', () => expect(previewText('hello')).toBe('hello'))
  it('truncates to maxLines with ellipsis', () => {
    const text = Array.from({ length: 10 }, (_, i) => `line ${i}`).join('\n')
    const out = previewText(text, 3)
    expect(out).toBe('line 0\nline 1\nline 2\n…')
  })
  it('truncates to maxChars', () => {
    const out = previewText('a'.repeat(400), 6, 10)
    expect(out).toBe('a'.repeat(10) + '…')
  })
})

describe('metaText', () => {
  it('returns "no metadata" for empty object', () => expect(metaText({})).toBe('no metadata'))
  it('formats entries joined by •', () => expect(metaText({ doc_type: 'contract', year: 2024 })).toBe('doc_type: contract • year: 2024'))
})

describe('schemaTypeStr', () => {
  it('handles $ref', () => expect(schemaTypeStr({ $ref: '#/definitions/Foo' })).toBe('Foo'))
  it('handles anyOf (filters null)', () => expect(schemaTypeStr({ anyOf: [{ type: 'string' }, { type: 'null' }] })).toBe('string'))
  it('handles array with item type', () => expect(schemaTypeStr({ type: 'array', items: { type: 'string' } })).toBe('list[string]'))
  it('returns plain type', () => expect(schemaTypeStr({ type: 'integer' })).toBe('integer'))
})

describe('simplifyExtractionSchemas', () => {
  it('leaves yaml without extraction_schema unchanged', () => {
    const yaml = 'name: test\nversion: 1'
    expect(simplifyExtractionSchemas(yaml)).toBe(yaml)
  })

  it('replaces inline JSON schema with field summary', () => {
    const schema = JSON.stringify({ properties: { name: { type: 'string', description: 'The name' } } })
    const yaml = `extraction_schema: '${schema}'`
    const out = simplifyExtractionSchemas(yaml)
    expect(out).toContain('# 1 fields')
    expect(out).toContain('#   name: string — The name')
    expect(out).not.toContain('"properties"')
  })

  it('ignores malformed JSON gracefully', () => {
    const yaml = "extraction_schema: 'not json'"
    expect(simplifyExtractionSchemas(yaml)).toBe(yaml)
  })
})

describe('streamSSE', () => {
  function makeResponse(lines) {
    const text = lines.join('\n') + '\n'
    const encoder = new TextEncoder()
    const encoded = encoder.encode(text)
    let pos = 0
    const stream = new ReadableStream({
      pull(controller) {
        if (pos >= encoded.length) { controller.close(); return }
        controller.enqueue(encoded.slice(pos, pos + 64))
        pos += 64
      },
    })
    return { body: stream }
  }

  it('yields parsed SSE events', async () => {
    const resp = makeResponse(['data: {"token":"hello"}', 'data: {"token":"world"}', 'data: [DONE]'])
    const results = []
    for await (const d of streamSSE(resp)) results.push(d)
    expect(results).toEqual([{ token: 'hello' }, { token: 'world' }])
  })

  it('skips non-data lines and malformed JSON', async () => {
    const resp = makeResponse([': comment', 'data: notjson', 'data: {"ok":true}'])
    const results = []
    for await (const d of streamSSE(resp)) results.push(d)
    expect(results).toEqual([{ ok: true }])
  })
})

describe('waitForTasks', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks() })

  it('resolves when all tasks complete', async () => {
    const fetchMock = vi.spyOn(global, 'fetch')
    fetchMock.mockResolvedValue({ ok: true, json: async () => ({ status: 'done', task_id: 't1', doc_id: 'doc1' }) })

    const promise = waitForTasks('http://localhost:8000', 'myapp', ['t1'], { pollInterval: 100 })
    await vi.runAllTimersAsync()
    const results = await promise
    expect(results[0].status).toBe('done')
    expect(results[0].doc_id).toBe('doc1')
  })

  it('marks tasks as failed on timeout', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValue({ ok: true, json: async () => ({ status: 'running' }) })
    const promise = waitForTasks('http://localhost:8000', 'myapp', ['t1'], { pollInterval: 100, timeout: 200 })
    await vi.runAllTimersAsync()
    const results = await promise
    expect(results[0].status).toBe('failed')
    expect(results[0].error).toBe('timeout')
  })

  it('calls onProgress as tasks complete', async () => {
    const onProgress = vi.fn()
    vi.spyOn(global, 'fetch')
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'pending', task_id: 't1' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'done', task_id: 't1', doc_id: 'd1' }) })
    const promise = waitForTasks('http://localhost:8000', 'app', ['t1'], { pollInterval: 100, onProgress })
    await vi.runAllTimersAsync()
    await promise
    expect(onProgress).toHaveBeenCalledWith(1, 1)
  })
})

describe('resolveArtifactLinks', () => {
  const API = 'http://localhost:8000'

  it('replaces the <app_name> placeholder and prefixes the API origin', () => {
    const out = resolveArtifactLinks(
      'Download: /applications/<app_name>/documents/lease-merged__27dd52c2.docx/download',
      API, 'my-app',
    )
    expect(out).toBe(
      'Download: http://localhost:8000/applications/my-app/documents/lease-merged__27dd52c2.docx/download',
    )
  })

  it('rewrites a relative path with a real app name onto the API origin', () => {
    const out = resolveArtifactLinks('/applications/lease-app/documents/x__ab12cd34.docx/download', API, 'cur')
    expect(out).toBe('http://localhost:8000/applications/lease-app/documents/x__ab12cd34.docx/download')
  })

  it('resolves the path inside a markdown link destination', () => {
    const out = resolveArtifactLinks('[merged](/applications/<app_name>/documents/f__aa.docx/download)', API, 'app')
    expect(out).toBe('[merged](http://localhost:8000/applications/app/documents/f__aa.docx/download)')
  })

  it('re-points an already-absolute URL at the configured API origin (idempotent)', () => {
    const abs = 'http://localhost:8000/applications/app/documents/f__aa.docx/download'
    expect(resolveArtifactLinks(abs, API, 'app')).toBe(abs)
  })

  it('encodes app names with unsafe characters', () => {
    const out = resolveArtifactLinks('/applications/<app_name>/documents/f__aa.docx/download', API, 'a b/c')
    expect(out).toBe('http://localhost:8000/applications/a%20b%2Fc/documents/f__aa.docx/download')
  })

  it('promotes a backtick-wrapped path to a labeled markdown link', () => {
    const out = resolveArtifactLinks(
      'Download it here:\n`/applications/<app_name>/documents/saas-001-amended__06467960.docx/download`',
      API, 'contract-analyst',
    )
    expect(out).toBe(
      'Download it here:\n[saas-001-amended.docx]' +
      '(http://localhost:8000/applications/contract-analyst/documents/saas-001-amended__06467960.docx/download)',
    )
  })

  it('lifts a backtick path without touching unrelated code spans', () => {
    const out = resolveArtifactLinks(
      'Merged `saas-001-amendment` into `saas-001`: `/applications/<app_name>/documents/x__aa00bb11.docx/download`',
      API, 'app',
    )
    expect(out).toContain('`saas-001-amendment`')
    expect(out).toContain('`saas-001`')
    expect(out).toContain('[x.docx](http://localhost:8000/applications/app/documents/x__aa00bb11.docx/download)')
  })

  it('leaves unrelated text and empty input untouched', () => {
    expect(resolveArtifactLinks('no links here', API, 'app')).toBe('no links here')
    expect(resolveArtifactLinks('', API, 'app')).toBe('')
    expect(resolveArtifactLinks(null, API, 'app')).toBe(null)
  })
})

describe('artifactLabel', () => {
  it('drops the disambiguating hex hash before the extension', () => {
    expect(artifactLabel('saas-001-amended__06467960.docx')).toBe('saas-001-amended.docx')
  })
  it('drops the hash when there is no extension', () => {
    expect(artifactLabel('report__06467960')).toBe('report')
  })
  it('only strips the trailing hash, keeping intraword double underscores', () => {
    expect(artifactLabel('my__file__06467960.docx')).toBe('my__file.docx')
  })
  it('leaves a plain filename untouched', () => {
    expect(artifactLabel('plain-name.docx')).toBe('plain-name.docx')
  })
})
